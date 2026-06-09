import numpy as np

from src.labels import CLASS_LABELS
from src.preprocessing import prepare_input


def predict_with_interpreter(interpreter, input_data: np.ndarray) -> dict:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    pred_idx = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    label = CLASS_LABELS[pred_idx] if pred_idx < len(CLASS_LABELS) else "Resultado desconocido"

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": output_data[0],
        "predicted_index": pred_idx,
    }


def predict_user_data(interpreter, preprocessor, datos_usuario: dict) -> dict:
    input_data = prepare_input(datos_usuario, preprocessor)
    return predict_with_interpreter(interpreter, input_data)