"""Kalkulator półmaratonu.

1. Użytkownik opisuje się w polu tekstowym.
2. LLM (OpenAI) wyłuskuje z opisu płeć, wiek i czas na 5 km do słownika (JSON).
3. Langfuse zapisuje każde wywołanie LLM i metryki skuteczności wyłuskiwania.
4. Model PyCaret, pobrany z Digital Ocean Spaces, przewiduje czas półmaratonu.
"""
import json
from datetime import date
from typing import Any

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import langfuse.openai  # noqa: F401 - sam import "podpina" OpenAI pod Langfuse: każde wywołanie trafia do Langfuse
from langfuse import get_client, observe
from openai import OpenAI
from pycaret.regression import load_model, predict_model

load_dotenv()   # wczytuje klucze z pliku .env

# --- Ustawienia -------------------------------------------------------------
BUCKET_NAME = "gotoitgosia13"
MODEL_PREFIX = "halfmarathon/models/"    # tu notebook zapisuje kolejne wersje modelu i latest.json
LOCAL_MODEL_NAME = "halfmarathon_model"  # PyCaret sam dopisuje .pkl
LLM_MODEL = "gpt-4o-mini"                # model OpenAI, który wyłuskuje dane z tekstu

FIELDS = {"gender": "płeć", "age": "wiek", "time_5km_seconds": "czas na 5 km"}

SYSTEM_PROMPT = f"""Wyłuskaj z tekstu dane biegacza i zwróć je jako JSON.
Jeśli jakiejś informacji nie ma w tekście, wpisz null - nie zgaduj.
- gender: "M" (mężczyzna) lub "K" (kobieta). Ustal płeć z tego, co napisano wprost,
  z form gramatycznych (np. "przebiegłam" -> "K", "przebiegłem" -> "M")
  albo z imienia (np. "Ania" -> "K", "Tomek" -> "M"). Jeśli imię może należeć
  do kobiety i do mężczyzny (np. "Sasza", "Alex", "Nikola"), wpisz null.
- age: wiek w latach.
  Jeśli wiek podano wprost (np. "50 lat", "mam 34 lata"), przepisz tę liczbę bez zmian - niczego nie przeliczaj.
  Tylko jeśli podano 4-cyfrowy rok urodzenia (np. "rocznik 1985", "urodziłam się w 1990"),
  policz: {date.today().year} - rok urodzenia.
- time_5km_seconds: czas na 5 km w sekundach, np. "25 minut" -> 1500, "24:30" -> 1470.
  Jeśli podano tempo na km (np. "5:00 min/km"), pomnóż je przez 5.

Przykłady:
"Marek, 41 lat, 5 km w 24 minuty" -> {{"gender": "M", "age": 41, "time_5km_seconds": 1440}}
"Ewa, rocznik 1985, 5 km biegam w 28:30" -> {{"gender": "K", "age": {date.today().year - 1985}, "time_5km_seconds": 1710}}
"Mam 60 lat i biegam 5 km w pół godziny" -> {{"gender": null, "age": 60, "time_5km_seconds": 1800}}"""

# Structured output: LLM musi zwrócić dokładnie te pola (null = brak informacji)
# Any = "dowolny typ" - biblioteka openai oczekuje tu własnego, bardzo szczegółowego typu
RESPONSE_FORMAT: Any = {
    "type": "json_schema",
    "json_schema": {
        "name": "runner_data",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gender": {"type": ["string", "null"], "enum": ["M", "K", None]},
                "age": {"type": ["integer", "null"]},
                "time_5km_seconds": {"type": ["integer", "null"]},
            },
            "required": ["gender", "age", "time_5km_seconds"],
            "additionalProperties": False,
        },
    },
}

openai_client = OpenAI()   # klucz bierze z OPENAI_API_KEY w .env
langfuse_client = get_client()   # klucze i adres bierze z LANGFUSE_* w .env


# --- Model z Digital Ocean Spaces --------------------------------------------
@st.cache_resource(ttl=3600)   # wczytuje model raz i trzyma w pamięci; co godzinę sprawdza, czy jest nowsza wersja
def get_model():
    s3 = boto3.client("s3")   # klucze i adres Spaces bierze z .env (tak jak w notebooku)

    # 1. latest.json mówi, która wersja modelu jest aktualna
    latest = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=f"{MODEL_PREFIX}latest.json")["Body"].read())
    version = latest["version"]

    # 2. pobranie i wczytanie tej wersji
    s3.download_file(Bucket=BUCKET_NAME, Key=f"{MODEL_PREFIX}{version}/{LOCAL_MODEL_NAME}.pkl",
                     Filename=f"{LOCAL_MODEL_NAME}.pkl")
    model: Any = load_model(LOCAL_MODEL_NAME)   # Any: edytor nie zna dokładnego typu pipeline'u PyCaret

    # 3. nazwa modelu = ostatni krok pipeline'u PyCaret (np. HuberRegressor)
    model_info = {"model_name": type(model.steps[-1][1]).__name__, "model_version": version}
    return model, model_info


# --- LLM + Langfuse ----------------------------------------------------------
@observe(name="extract-runner-data")   # cała funkcja = jeden trace w Langfuse
def extract_data(text, model_info):
    # Nazwa i wersja modelu zapisane przy trace - w Langfuse widać, który model obsłużył zapytanie
    langfuse_client.update_current_span(metadata=model_info)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format=RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content or "{}"   # pusta odpowiedź = brak wszystkich danych
    data = json.loads(content)
    missing = [label for key, label in FIELDS.items() if data.get(key) is None]

    # Metryki skuteczności LLM-a, widoczne w Langfuse przy tym trace
    found = [key for key in FIELDS if data.get(key) is not None]   # pola, które LLM odczytał
    langfuse_client.score_current_trace(name="extraction_complete", value=0 if missing else 1)
    langfuse_client.score_current_trace(
        name="data_completeness",
        value=len(found) / len(FIELDS),
        comment=f"Fields found: {found}",
    )

    return data, missing


def format_time(seconds):
    """1500 -> '25:00', 6553 -> '1:49:13'"""
    h, rest = divmod(int(round(seconds)), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --- Interfejs ---------------------------------------------------------------
st.title("🏃 Half Marathon Predictor")
st.caption("Narzędzie oparte na modelu AI")

st.write("Chcesz wiedzieć, ile czasu zajmie Ci przebiegnięcie półmaratonu?")
st.markdown("""
1. Przedstaw się i podaj swoją **płeć**, **wiek** oraz **czas na 5 km**.
2. Kliknij **„Oszacuj mój czas”**.
3. Model AI odczyta dane z Twojego opisu i oszacuje Twój czas na mecie.
""")
st.caption("Szacunek opiera się na ok. 18 tys. wyników Półmaratonu Wrocławskiego (2023–2024). "
           "Typowa różnica między szacunkiem a rzeczywistym wynikiem to ok. 5 minut.")

text = st.text_area("Opowiedz o sobie",
                    placeholder="Np. Cześć, jestem Ania, mam 34 lata, a 5 km przebiegłam w 27 minut.")

if st.button("Oszacuj mój czas", type="primary"):
    with st.spinner("Analizuję Twój opis..."):
        model, model_info = get_model()
        data, missing = extract_data(text, model_info)
        langfuse_client.flush()   # wysyła dane do Langfuse od razu

    with st.expander("Co zrozumiałem z Twojego opisu"):
        gender = {"M": "mężczyzna", "K": "kobieta"}.get(data.get("gender"), "—")
        age = data.get("age")
        time_5km = data.get("time_5km_seconds")
        st.write(f"- **Płeć:** {gender}")
        st.write(f"- **Wiek:** {age if age is not None else '—'}")
        st.write(f"- **Czas na 5 km:** {format_time(time_5km) if time_5km is not None else '—'}")

    if missing:
        st.warning(f"Brakuje mi danych: **{', '.join(missing)}**. Uzupełnij opis i spróbuj ponownie.")
    else:
        # Kolumny muszą się nazywać tak samo jak przy trenowaniu modelu
        runner = pd.DataFrame([{
            "gender": data["gender"],
            "age": data["age"],
            "5km_time": data["time_5km_seconds"],
        }])
        prediction = predict_model(model, data=runner, verbose=False)["prediction_label"].iloc[0]

        st.metric("Przewidywany czas na mecie", format_time(prediction))
        st.info("Model uczył się na czasach z pierwszych 5 km półmaratonu. Jeśli podałeś czas "
                "z osobnego biegu na 5 km, Twój rzeczywisty wynik może być wolniejszy.")
        st.caption(f"LLM: {LLM_MODEL} · model predykcji: {model_info['model_name']} "
                   f"(wersja {model_info['model_version']})")
