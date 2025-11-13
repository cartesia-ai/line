def is_e164_phone_number(phone: str) -> bool:
    """Check if a string is a valid E.164 compliant phone number.

    E.164 format requirements:
    - Must start with '+'
    - Followed by 5-15 digits
    - No spaces, hyphens, or other characters

    Args:
        phone: The phone number string to validate

    Returns:
        bool: True if the string is E.164 compliant, False otherwise


    Note: 1+4=5 is practically the mininum number of digits. A country can have
    a short national phone number code (len=4) if they are small (e.g. Falkland Islands)
    """
    # Must start with '+'
    if not phone.startswith("+"):
        return False

    # Remove the '+' and check the rest
    digits = phone[1:]

    # Must be between 1 and 15 digits
    if not digits.isdigit() or len(digits) < 5 or len(digits) > 15:
        return False

    return True
