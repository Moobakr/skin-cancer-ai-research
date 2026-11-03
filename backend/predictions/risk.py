RISK_MAP = {
    "MEL": "High risk",
    "BCC": "High risk",
    "SCC": "High risk",
    "BOD": "High risk",
    "ACK": "Medium risk",
    "AK": "Medium risk",
    "SEK": "Lower risk",
    "NEV": "Lower risk",
}

CLASS_DESCRIPTIONS = {
    "ACK": "Actinic keratosis is commonly treated as a precancerous or sun-damage related lesion category.",
    "BCC": "Basal cell carcinoma is a common skin cancer category and should be evaluated professionally.",
    "SCC": "Squamous cell carcinoma is a skin cancer category and should be evaluated professionally.",
    "BOD": "Bowen's disease is commonly considered an in-situ squamous cell carcinoma category.",
    "MEL": "Melanoma is a serious skin cancer category where professional evaluation is especially important.",
    "NEV": "Nevus is a mole-like lesion category, but suspicious or changing lesions still require medical review.",
    "SEK": "Seborrheic keratosis is often benign, but image-based AI cannot confirm diagnosis.",
}

DISCLAIMER = (
    "This AI result is for educational screening support only and is not a medical diagnosis. "
    "Please consult a dermatologist for professional evaluation."
)


def risk_for_class(class_name: str) -> str:
    return RISK_MAP.get(str(class_name).upper(), "Unknown risk")


def description_for_class(class_name: str) -> str:
    return CLASS_DESCRIPTIONS.get(str(class_name).upper(), "No class description is available for this label.")


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


def uncertainty_warnings(probabilities: dict[str, float]) -> list[str]:
    if not probabilities:
        return ["No probabilities were returned by the model."]
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    warnings = []
    top_class, top_prob = sorted_probs[0]
    if top_prob < 0.50:
        warnings.append("The model confidence is low. Treat this result as highly uncertain.")
    if len(sorted_probs) > 1:
        margin = sorted_probs[0][1] - sorted_probs[1][1]
        if margin < 0.12:
            warnings.append(
                f"The top two classes are close ({sorted_probs[0][0]} and {sorted_probs[1][0]}), so the prediction is uncertain."
            )
    return warnings
