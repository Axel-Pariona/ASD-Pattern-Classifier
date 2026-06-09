from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "modelo_autismo.tflite"
DATASET_PATH = ROOT_DIR / "data" / "dataset_clinico_autismo.csv"

CATEGORICAL_COLS = [
    "Sexo",
    "Lenguaje",
    "Comunicación no verbal",
    "Contacto visual",
    "Interacción social",
    "Respuesta al nombre",
    "Estereotipias",
    "Intereses restringidos",
    "Regulación emocional",
    "TDAH",
    "Discapacidad intelectual",
    "Hipersensibilidad sensorial",
    "Trastornos del sueño",
    "Alimentación selectiva",
    "Antecedentes familiares",
]

NUMERIC_COLS = [
    "Edad (meses)",
    "Puntaje riesgo",
]

FEATURE_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS