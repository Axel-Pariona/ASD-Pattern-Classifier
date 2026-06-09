from src.labels import RISK_LEVELS


def calculate_risk_score(datos: dict) -> int:
    score = 0

    if datos["Lenguaje"] == "No verbal":
        score += 3
    elif datos["Lenguaje"] == "Ecolalia":
        score += 2
    elif datos["Lenguaje"] == "Frases simples":
        score += 1

    if datos["Comunicación no verbal"] == "Ausente":
        score += 3
    elif datos["Comunicación no verbal"] == "Muy limitada":
        score += 2
    elif datos["Comunicación no verbal"] == "Limitada":
        score += 1

    if datos["Contacto visual"] == "Evitativo":
        score += 2
    elif datos["Contacto visual"] == "Intermitente":
        score += 1

    if datos["Interacción social"] == "Ausente":
        score += 3
    elif datos["Interacción social"] == "Pasiva":
        score += 2
    elif datos["Interacción social"] == "Inapropiada":
        score += 1

    if datos["Respuesta al nombre"] == "Nunca":
        score += 2
    elif datos["Respuesta al nombre"] == "A veces":
        score += 1

    if datos["Estereotipias"] == "Muy frecuentes":
        score += 3
    elif datos["Estereotipias"] == "Frecuentes":
        score += 2
    elif datos["Estereotipias"] == "Ocasionales":
        score += 1

    if datos["Intereses restringidos"] == "Muy intensos":
        score += 2
    elif datos["Intereses restringidos"] == "Persistentes":
        score += 1

    if datos["Regulación emocional"] == "Autolesiva":
        score += 3
    elif datos["Regulación emocional"] == "Crisis frecuentes":
        score += 2
    elif datos["Regulación emocional"] == "Ocasionales":
        score += 1

    if datos["Discapacidad intelectual"] == "Sí":
        score += 2

    if datos["TDAH"] == "Sí":
        score += 1

    return score


def interpret_risk_score(score: int) -> dict:
    for item in RISK_LEVELS:
        if score >= item["min_score"]:
            return item

    return {
        "level": "No determinado",
        "icon": "⚪",
    }