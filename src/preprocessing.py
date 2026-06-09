import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


OPCIONES_VARIABLES = {
    "Sexo": ["Masculino", "Femenino"],
    "Lenguaje": ["No verbal", "Ecolalia", "Frases simples", "Lenguaje funcional"],
    "Comunicación no verbal": ["Ausente", "Muy limitada", "Limitada", "Adecuada"],
    "Contacto visual": ["Evitativo", "Intermitente", "Sostenido", "Natural"],
    "Interacción social": ["Ausente", "Pasiva", "Inapropiada", "Adecuada"],
    "Respuesta al nombre": ["Nunca", "A veces", "Siempre"],
    "Estereotipias": ["Muy frecuentes", "Frecuentes", "Ocasionales", "Ausentes"],
    "Intereses restringidos": ["Muy intensos", "Persistentes", "Leves", "Ausentes"],
    "Regulación emocional": ["Autolesiva", "Crisis frecuentes", "Ocasionales", "Adecuada"],
    "TDAH": ["Sí", "No"],
    "Discapacidad intelectual": ["Sí", "No"],
    "Hipersensibilidad sensorial": ["Alta", "Moderada", "Leve", "Ninguna"],
    "Trastornos del sueño": ["Severo", "Moderado", "Leve", "Normal"],
    "Alimentación selectiva": ["Alta", "Moderada", "Leve", "Ninguna"],
    "Antecedentes familiares": ["TEA", "TDAH", "Discapacidad intelectual", "Ninguno"],
}


def build_reference_dataframe(samples: int = 500) -> pd.DataFrame:
    """
    Construye un DataFrame de referencia para ajustar el preprocesador.

    Este proyecto no conserva el preprocessor entrenado original, por lo que se
    reconstruye una base de referencia con todas las categorías posibles para
    mantener la dimensión esperada por el modelo.
    """
    rows = []

    for _ in range(samples):
        row = {
            "Edad (meses)": np.random.randint(3, 37),
            "Puntaje riesgo": np.random.randint(0, 25),
        }

        for col, options in OPCIONES_VARIABLES.items():
            row[col] = np.random.choice(options)

        rows.append(row)

    for variable, options in OPCIONES_VARIABLES.items():
        for option in options:
            row = {
                "Edad (meses)": 24,
                "Puntaje riesgo": 10,
            }

            for col, col_options in OPCIONES_VARIABLES.items():
                row[col] = col_options[0]

            row[variable] = option
            rows.append(row)

    return pd.DataFrame(rows)


def create_preprocessor() -> ColumnTransformer:
    """
    Crea y ajusta el preprocesador usado antes de enviar los datos al modelo.

    - OneHotEncoder para variables categóricas.
    - StandardScaler para variables numéricas.
    """
    reference_df = build_reference_dataframe()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )

    preprocessor.fit(reference_df)

    return preprocessor


def prepare_input(datos_usuario: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Convierte los datos ingresados por el usuario en un arreglo float32 compatible
    con TensorFlow Lite.
    """
    df_usuario = pd.DataFrame([datos_usuario])
    processed = preprocessor.transform(df_usuario)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    return processed.astype(np.float32)