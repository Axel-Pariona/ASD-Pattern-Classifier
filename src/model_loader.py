import tensorflow as tf

from src.config import MODEL_PATH


def load_tflite_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    return interpreter