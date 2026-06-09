import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.model_loader import load_tflite_model
from src.preprocessing import OPCIONES_VARIABLES, create_preprocessor
from src.predictor import predict_user_data
from src.risk_score import calculate_risk_score, interpret_risk_score
from src.labels import CLASS_LABELS


st.set_page_config(
    page_title="Predictor orientativo de TEA",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def get_model():
    return load_tflite_model()


@st.cache_resource
def get_preprocessor():
    return create_preprocessor()


def main():
    st.title("Predictor orientativo de patrones asociados al TEA")
    st.markdown("### Aplicación académica de apoyo basada en Machine Learning")
    st.markdown(
        "**Modelo:** Red neuronal convertida a TensorFlow Lite | "
        "**Uso:** clasificación orientativa con variables clínicas y conductuales"
    )

    st.warning(
        "⚠️ Este proyecto es una herramienta académica de apoyo orientativo. "
        "No sustituye una evaluación clínica profesional ni debe usarse como diagnóstico definitivo."
    )

    try:
        interpreter = get_model()
        preprocessor = get_preprocessor()
    except Exception as e:
        st.error(f"No se pudo cargar el modelo o el preprocesador: {e}")
        st.stop()

    st.sidebar.header("Información del modelo")
    st.sidebar.markdown(
        """
        **Tipo:** Red neuronal  
        **Formato:** TensorFlow Lite  
        **Variables:** 17 características  
        **Clases:** 5 categorías  
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Modo de predicción")

    modo = st.sidebar.radio(
        "Selecciona el modo:",
        [
            "Evaluación completa",
            "Datos simulados",
            "Información del modelo",
        ],
    )

    if modo == "Evaluación completa":
        mostrar_evaluacion_completa(interpreter, preprocessor)
    elif modo == "Datos simulados":
        mostrar_datos_simulados(interpreter, preprocessor)
    else:
        mostrar_informacion_modelo()


def mostrar_evaluacion_completa(interpreter, preprocessor):
    st.header("Evaluación completa")
    st.markdown("Completa los campos para obtener una clasificación orientativa.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Información básica")
        edad = st.slider("Edad (en meses)", 3, 36, 36)
        sexo = st.selectbox("Sexo", OPCIONES_VARIABLES["Sexo"])

        st.subheader("Comunicación y lenguaje")
        lenguaje = st.selectbox("Nivel de lenguaje", OPCIONES_VARIABLES["Lenguaje"])
        comunicacion_nv = st.selectbox(
            "Comunicación no verbal",
            OPCIONES_VARIABLES["Comunicación no verbal"],
        )
        contacto_visual = st.selectbox(
            "Contacto visual",
            OPCIONES_VARIABLES["Contacto visual"],
        )
        respuesta_nombre = st.selectbox(
            "Respuesta al nombre",
            OPCIONES_VARIABLES["Respuesta al nombre"],
        )

        st.subheader("Interacción social")
        interaccion_social = st.selectbox(
            "Interacción social",
            OPCIONES_VARIABLES["Interacción social"],
        )

        st.subheader("Comportamientos repetitivos")
        estereotipias = st.selectbox(
            "Estereotipias",
            OPCIONES_VARIABLES["Estereotipias"],
        )
        intereses_restringidos = st.selectbox(
            "Intereses restringidos",
            OPCIONES_VARIABLES["Intereses restringidos"],
        )

    with col2:
        st.subheader("Regulación emocional")
        regulacion = st.selectbox(
            "Regulación emocional",
            OPCIONES_VARIABLES["Regulación emocional"],
        )

        st.subheader("Comorbilidades")
        tdah = st.selectbox("TDAH", OPCIONES_VARIABLES["TDAH"])
        discapacidad_int = st.selectbox(
            "Discapacidad intelectual",
            OPCIONES_VARIABLES["Discapacidad intelectual"],
        )

        st.subheader("Aspectos sensoriales")
        hipersensibilidad = st.selectbox(
            "Hipersensibilidad sensorial",
            OPCIONES_VARIABLES["Hipersensibilidad sensorial"],
        )

        st.subheader("Hábitos")
        sueno = st.selectbox(
            "Trastornos del sueño",
            OPCIONES_VARIABLES["Trastornos del sueño"],
        )
        alimentacion = st.selectbox(
            "Alimentación selectiva",
            OPCIONES_VARIABLES["Alimentación selectiva"],
        )

        st.subheader("Antecedentes")
        antecedentes = st.selectbox(
            "Antecedentes familiares",
            OPCIONES_VARIABLES["Antecedentes familiares"],
        )

    st.markdown("---")

    if st.button("Realizar clasificación orientativa", type="primary", use_container_width=True):
        datos_usuario = {
            "Edad (meses)": edad,
            "Sexo": sexo,
            "Lenguaje": lenguaje,
            "Comunicación no verbal": comunicacion_nv,
            "Contacto visual": contacto_visual,
            "Interacción social": interaccion_social,
            "Respuesta al nombre": respuesta_nombre,
            "Estereotipias": estereotipias,
            "Intereses restringidos": intereses_restringidos,
            "Regulación emocional": regulacion,
            "TDAH": tdah,
            "Discapacidad intelectual": discapacidad_int,
            "Hipersensibilidad sensorial": hipersensibilidad,
            "Trastornos del sueño": sueno,
            "Alimentación selectiva": alimentacion,
            "Antecedentes familiares": antecedentes,
        }

        procesar_prediccion(interpreter, preprocessor, datos_usuario)


def mostrar_datos_simulados(interpreter, preprocessor):
    st.header("Predicción con datos simulados")
    st.info("Esta opción utiliza valores predeterminados para probar el flujo del modelo.")

    if st.button("Generar predicción simulada", type="primary"):
        datos_simulados = {
            "Edad (meses)": 30,
            "Sexo": "Masculino",
            "Lenguaje": "Frases simples",
            "Comunicación no verbal": "Limitada",
            "Contacto visual": "Intermitente",
            "Interacción social": "Pasiva",
            "Respuesta al nombre": "A veces",
            "Estereotipias": "Frecuentes",
            "Intereses restringidos": "Persistentes",
            "Regulación emocional": "Ocasionales",
            "TDAH": "No",
            "Discapacidad intelectual": "No",
            "Hipersensibilidad sensorial": "Moderada",
            "Trastornos del sueño": "Leve",
            "Alimentación selectiva": "Moderada",
            "Antecedentes familiares": "TEA",
        }

        st.subheader("Datos utilizados")

        col1, col2 = st.columns(2)
        items = list(datos_simulados.items())
        midpoint = len(items) // 2

        with col1:
            for key, value in items[:midpoint]:
                st.write(f"**{key}:** {value}")

        with col2:
            for key, value in items[midpoint:]:
                st.write(f"**{key}:** {value}")

        procesar_prediccion(interpreter, preprocessor, datos_simulados)


def procesar_prediccion(interpreter, preprocessor, datos_usuario):
    try:
        puntaje_riesgo = calculate_risk_score(datos_usuario)
        datos_usuario["Puntaje riesgo"] = puntaje_riesgo

        with st.spinner("Procesando información..."):
            prediction = predict_user_data(interpreter, preprocessor, datos_usuario)

        resultado = prediction["label"]
        confianza = prediction["confidence"]
        probabilidades = prediction["probabilities"]

        mostrar_resultados(resultado, confianza, probabilidades, puntaje_riesgo)

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.info("Revisa que el modelo, el preprocesador y las dimensiones de entrada sean compatibles.")


def mostrar_resultados(resultado, confianza, probabilidades, puntaje_riesgo):
    st.markdown("---")
    st.header("Resultados orientativos")

    risk_info = interpret_risk_score(puntaje_riesgo)
    riesgo_nivel = risk_info["level"]
    riesgo_icono = risk_info["icon"]

    col1, col2, col3 = st.columns(3)

    with col1:
        if resultado == "Desarrollo típico":
            st.success(f"**Clasificación:** {resultado}")
        elif "TEA" in resultado:
            st.warning(f"**Clasificación:** {resultado}")
        else:
            st.info(f"**Clasificación:** {resultado}")

    with col2:
        st.metric("Confianza del modelo", f"{confianza * 100:.1f}%")

    with col3:
        st.metric("Puntaje de riesgo", f"{puntaje_riesgo}/24", f"{riesgo_icono} {riesgo_nivel}")

    if len(probabilidades) > 0:
        st.subheader("Distribución de probabilidades")

        chart_data = {
            CLASS_LABELS[i]: float(probabilidades[i])
            for i in range(min(len(CLASS_LABELS), len(probabilidades)))
        }

        st.bar_chart(chart_data)

        df_prob = pd.DataFrame(
            list(chart_data.items()),
            columns=["Categoría", "Probabilidad"],
        )
        df_prob["Probabilidad"] = df_prob["Probabilidad"].apply(lambda x: f"{x * 100:.1f}%")

        st.dataframe(df_prob, use_container_width=True)

    st.warning(
        "⚠️ Los resultados son orientativos y no sustituyen una evaluación clínica profesional. "
        "Ante cualquier duda, se recomienda consultar con un especialista."
    )


def mostrar_informacion_modelo():
    st.header("Información del modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Características generales")
        st.markdown(
            """
            - **Tipo:** Red neuronal
            - **Formato:** TensorFlow Lite
            - **Variables de entrada:** 17 características
            - **Categorías de salida:** 5
            - **Uso:** proyecto académico de clasificación orientativa
            """
        )

        st.subheader("Alcance")
        st.markdown(
            """
            - Proyecto de Machine Learning aplicado.
            - Uso de datos clínicos y conductuales.
            - Inferencia mediante modelo `.tflite`.
            - Interfaz interactiva con Streamlit.
            """
        )

    with col2:
        st.subheader("Variables utilizadas")
        st.markdown(
            """
            **Información básica**
            - Edad
            - Sexo

            **Comunicación**
            - Lenguaje
            - Comunicación no verbal
            - Contacto visual
            - Respuesta al nombre

            **Interacción y conducta**
            - Interacción social
            - Estereotipias
            - Intereses restringidos
            - Regulación emocional

            **Comorbilidades y antecedentes**
            - TDAH
            - Discapacidad intelectual
            - Hipersensibilidad sensorial
            - Trastornos del sueño
            - Alimentación selectiva
            - Antecedentes familiares
            """
        )

    st.info(
        "Este modelo fue desarrollado con fines académicos. "
        "Debe interpretarse como una herramienta de apoyo orientativo, no como diagnóstico médico."
    )


if __name__ == "__main__":
    main()

