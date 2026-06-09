# Autism Prediction

Autism Prediction es un proyecto académico de Machine Learning orientado a la clasificación aproximada de patrones asociados al Trastorno del Espectro Autista (TEA), utilizando variables clínicas y conductuales simuladas.

El proyecto incluye un modelo de red neuronal convertido a TensorFlow Lite, una interfaz interactiva desarrollada con Streamlit y versiones alternativas de interfaz en Tkinter, PyQt y Kivy.

> **Nota importante:** Este proyecto fue desarrollado con fines académicos. No sustituye una evaluación clínica profesional ni debe utilizarse como herramienta de diagnóstico médico definitivo.

## Objetivo del proyecto

El objetivo principal es aplicar técnicas de Machine Learning para construir una aplicación de apoyo orientativo que permita ingresar variables clínicas y conductuales, procesarlas y obtener una clasificación aproximada mediante un modelo previamente entrenado.

El proyecto busca demostrar:

* Preparación de datos clínicos simulados.
* Entrenamiento de un modelo de clasificación.
* Conversión del modelo a TensorFlow Lite.
* Uso de inferencia local.
* Construcción de una interfaz interactiva.
* Organización modular del código.
* Separación entre interfaz, preprocesamiento, modelo y predicción.

## Tecnologías utilizadas

* Python
* Streamlit
* TensorFlow Lite
* Pandas
* NumPy
* Scikit-learn
* Tkinter
* PyQt
* Kivy
* Jupyter Notebook

## Características principales

* Interfaz principal con Streamlit.
* Modelo en formato `.tflite`.
* Predicción basada en 17 variables clínicas y conductuales.
* Cálculo de puntaje de riesgo orientativo.
* Visualización de probabilidades por categoría.
* Modo de evaluación completa.
* Modo de datos simulados.
* Información general del modelo.
* Código reorganizado en módulos reutilizables.

## Estructura del proyecto

```txt
Autism_prediction/
  app/
    streamlit_app.py
    tkinter_app.py
    pyqt_app.py
    kivy_app.py

  data/
    dataset_clinico_autismo.csv

  models/
    modelo_autismo.tflite

  notebooks/
    mark3.ipynb

  src/
    __init__.py
    config.py
    labels.py
    model_loader.py
    predictor.py
    preprocessing.py
    risk_score.py

  tests/
    debug_dimensions.py
    test_app.py

  .gitignore
  README.md
  requirements.txt
  runtime.txt
```

## Descripción de carpetas

### `app/`

Contiene las interfaces de usuario del proyecto.

* `streamlit_app.py`: aplicación principal con interfaz web local.
* `tkinter_app.py`: versión alternativa con Tkinter.
* `pyqt_app.py`: versión alternativa con PyQt.
* `kivy_app.py`: versión alternativa con Kivy.

### `src/`

Contiene la lógica reutilizable del proyecto.

* `config.py`: define rutas principales y columnas del modelo.
* `labels.py`: contiene etiquetas de clasificación y niveles de riesgo.
* `model_loader.py`: carga el modelo TensorFlow Lite.
* `predictor.py`: ejecuta la inferencia con el modelo.
* `preprocessing.py`: transforma los datos ingresados al formato esperado por el modelo.
* `risk_score.py`: calcula e interpreta el puntaje de riesgo orientativo.

### `models/`

Contiene el modelo entrenado convertido a TensorFlow Lite.

```txt
modelo_autismo.tflite
```

### `data/`

Contiene el dataset utilizado en el proyecto.

```txt
dataset_clinico_autismo.csv
```

### `notebooks/`

Contiene el notebook usado para el análisis, entrenamiento y generación del modelo.

```txt
mark3.ipynb
```

### `tests/`

Contiene scripts auxiliares de prueba y depuración.

## Variables utilizadas

El modelo utiliza variables relacionadas con:

* Edad.
* Sexo.
* Lenguaje.
* Comunicación no verbal.
* Contacto visual.
* Interacción social.
* Respuesta al nombre.
* Estereotipias.
* Intereses restringidos.
* Regulación emocional.
* TDAH.
* Discapacidad intelectual.
* Hipersensibilidad sensorial.
* Trastornos del sueño.
* Alimentación selectiva.
* Antecedentes familiares.
* Puntaje de riesgo.

## Categorías de salida

El modelo clasifica los registros en cinco categorías:

* Desarrollo típico.
* TEA - Nivel 1.
* TEA - Nivel 2.
* TEA - Nivel 3.
* Indeterminado.

Estas categorías se presentan únicamente con fines académicos y orientativos.

## Instalación

Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd Autism_prediction
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno virtual en Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Activar el entorno virtual en Linux o WSL:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar la aplicación principal con Streamlit:

```bash
streamlit run app/streamlit_app.py
```

También se pueden ejecutar interfaces alternativas:

```bash
python app/tkinter_app.py
```

```bash
python app/pyqt_app.py
```

```bash
python app/kivy_app.py
```

## Flujo general

```txt
Datos ingresados por el usuario
  ↓
Cálculo de puntaje de riesgo
  ↓
Preprocesamiento de variables
  ↓
Modelo TensorFlow Lite
  ↓
Predicción
  ↓
Visualización de resultados en Streamlit
```

## Alcance del proyecto

El proyecto implementa una aplicación académica funcional de clasificación orientativa.

Incluye:

* Modelo entrenado y convertido a TensorFlow Lite.
* Interfaz principal en Streamlit.
* Preprocesamiento de variables categóricas y numéricas.
* Inferencia local.
* Visualización de probabilidades.
* Separación modular del código.
* Versiones alternativas de interfaz.

## Limitaciones

* No sustituye una evaluación clínica profesional.
* No debe utilizarse como diagnóstico médico definitivo.
* El dataset utilizado corresponde a un contexto académico/simulado.
* El preprocesador se reconstruye en tiempo de ejecución.
* No se conserva un pipeline entrenado serializado.
* No se implementa validación clínica real.
* No se optimizó para producción.
* Las interfaces alternativas pueden requerir ajustes adicionales.

## Posibles mejoras

* Guardar y reutilizar el preprocesador entrenado.
* Mejorar el entrenamiento del modelo.
* Agregar métricas detalladas de evaluación.
* Implementar pruebas automatizadas.
* Mejorar la validación de entradas.
* Optimizar dependencias para despliegue.
* Crear una versión ligera solo con TensorFlow Lite Runtime.
* Documentar mejor el proceso de entrenamiento.
* Agregar capturas de pantalla de la aplicación.

## Estado del proyecto

Proyecto académico funcional y reorganizado para fines de presentación en GitHub.

No se encuentra orientado a producción ni a uso clínico real.

## Autor

Desarrollado por Axel Pariona como proyecto académico de Machine Learning aplicado.
