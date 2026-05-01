import difflib


def smart_match(input_str: str, choices: list[str]) -> list[str]:
    """Find the best matching string from choices using exact, substring, then fuzzy matching."""
    input_str = input_str.lower().strip()

    if input_str in choices:
        return [input_str]

    contains_matches = [c for c in choices if input_str in c]
    if contains_matches:
        return contains_matches[:5]

    return difflib.get_close_matches(input_str, choices, n=5, cutoff=0.3)
