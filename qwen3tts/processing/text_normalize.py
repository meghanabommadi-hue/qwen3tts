"""Text normalization before TTS synthesis.

Handles (English-first, extensible for other languages):
  - Currency amounts → spoken words:  $5,000 → "five thousand"
  - Plain digit sequences → spoken words:  12345 → "twelve thousand three hundred forty five"
  - Abbreviation expansion: Mr. → mister, etc.
  - Punctuation cleanup for Qwen3-TTS input formatting
"""

import re

ALWAYS_EXPAND = {
    "mr.": "mister",
    "mrs.": "missus",
    "ms.": "miss",
    "dr.": "doctor",
    "prof.": "professor",
    "sr.": "senior",
    "jr.": "junior",
    "hon.": "honorable",
    "rev.": "reverend",

    "ltd.": "limited",
    "pvt.": "private",
    "inc.": "incorporated",
    "corp.": "corporation",
    "co.": "company",

    "rd.": "road",
    "st.": "street",
    "ave.": "avenue",
    "blvd.": "boulevard",
    "ln.": "lane",
    "apt.": "apartment",
    "no.": "number",

    "amt": "amount",
    "a/c": "account",
    "bal.": "balance",
    "min.": "minimum",
    "max.": "maximum",
    "dept.": "department",

    "etc.": "etcetera",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
}

# Pre-compile abbreviation patterns once at import time
_ABBR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE), full)
    for abbr, full in ALWAYS_EXPAND.items()
]

# ---------------------------------------------------------------------------
# English number-to-words
# ---------------------------------------------------------------------------

_EN_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]


def _num_to_english(n: int) -> str:
    """Convert a non-negative integer to English words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _num_to_english(-n)
    parts = []
    if n >= 1_000_000_000:
        parts.append(_num_to_english(n // 1_000_000_000) + " billion")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_num_to_english(n // 1_000_000) + " million")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(_num_to_english(n // 1_000) + " thousand")
        n %= 1_000
    if n >= 100:
        parts.append(_EN_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        t = _EN_TENS[n // 10]
        o = _EN_ONES[n % 10]
        parts.append(t + (" " + o if o else ""))
    elif n > 0:
        parts.append(_EN_ONES[n])
    return " ".join(parts)


# Currency: $5,000 / £100 / €200 / "dollars 500" etc.
_EN_CURRENCY_RE = re.compile(
    r'(?:[$£€¥]\s*|(?:dollars?|pounds?|euros?|yen)\s+)'
    r'([0-9][0-9,]*)',
    re.IGNORECASE,
)
_EN_PLAIN_NUM_RE = re.compile(r'[0-9]+')


def _replace_en_currency(m: re.Match) -> str:
    digits = m.group(1).replace(",", "")
    return _num_to_english(int(digits))


def _replace_en_plain(m: re.Match) -> str:
    return _num_to_english(int(m.group(0)))


def normalize_text(text: str) -> str:
    """Normalize text before passing to the Qwen3 TTS synthesizer.

    - Expands currency amounts to English words
    - Expands digit sequences to English words
    - Expands common abbreviations
    - Cleans up punctuation for cleaner prosody
    """
    # Normalize trailing punctuation for better prosody
    text = text.replace("।", " .")  # Hindi full stop → ASCII period (multilingual support)
    idx = text.rfind(",")
    if idx != -1:
        text = text[:idx] + " ,.." + text[idx + 1:]
    if text.endswith("."):
        text = text[:-1] + " ..."

    # Currency amounts → English words
    text = _EN_CURRENCY_RE.sub(_replace_en_currency, text)
    # Plain digits → English words
    text = _EN_PLAIN_NUM_RE.sub(_replace_en_plain, text)
    # Abbreviations
    for pattern, full in _ABBR_PATTERNS:
        text = pattern.sub(full, text)

    return text
