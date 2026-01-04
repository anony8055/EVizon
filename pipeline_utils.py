def clean_event_name(name: str) -> str:
    if not name:
        return "Unknown"
    name = name.strip()
    replacements = {
        "coni": "Convocation",
        "fest kolkata": "DevFest Kolkata",
        "marathon": "Kolkata Marathon",
        "devfest": "DevFest",
    }
    lower = name.lower()
    for k, v in replacements.items():
        if k in lower:
            return v
    return name.title()
