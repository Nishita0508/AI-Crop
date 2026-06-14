"""
utils/validators.py
Input validation helpers for forms and API inputs.
"""

import re

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+\@[\w\-]+\.[a-zA-Z]{2,}$")


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_REGEX.match(email))


def is_valid_username(username: str) -> bool:
    return bool(username) and 3 <= len(username) <= 30 and username.isalnum()


def is_valid_password(password: str) -> bool:
    return bool(password) and len(password) >= 6


def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def validate_crop_input(data: dict):
    """
    Validate crop recommendation input fields.
    Returns (is_valid: bool, errors: list, parsed: dict)
    """
    required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    errors = []
    parsed = {}

    for field in required_fields:
        value = data.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"'{field}' is required.")
            continue
        try:
            parsed[field] = float(value)
        except ValueError:
            errors.append(f"'{field}' must be a number.")

    # Range sanity checks
    if "ph" in parsed and not (0 <= parsed["ph"] <= 14):
        errors.append("'ph' must be between 0 and 14.")
    if "humidity" in parsed and not (0 <= parsed["humidity"] <= 100):
        errors.append("'humidity' must be between 0 and 100.")

    return (len(errors) == 0, errors, parsed)
