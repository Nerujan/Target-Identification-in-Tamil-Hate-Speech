import re
import unicodedata

# Synchronous translator (deep-translator)
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    translator = GoogleTranslator(source="auto", target="ta")
except Exception:
    TRANSLATOR_AVAILABLE = False
    translator = None

# Optional: Aksharamukha for transliteration (best if available)
try:
    from aksharamukha import transliterate as ak_trans
    AK_AVAILABLE = True
except Exception:
    AK_AVAILABLE = False

# Optional: language detection (not required)
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# Regex helpers
RE_WORD = re.compile(r"[A-Za-z]+")
RE_TAMIL_CHAR = re.compile(r"[\u0B80-\u0BFF]+")
# used by cleaning to remove ASCII blobs
_NON_TAMIL = re.compile(r"[A-Za-z0-9\.\,\:\;\+\=\(\)\[\]\{\}<>@#\$%\^&\*_\"\/\\\|~`]+")

# Patterns for removing superscripts / odd glyphs from transliteration output
_SUPER_SCRIPT_RE = re.compile(r"[\u00B2\u00B3\u00B9\u2070-\u2079\u02BC\u02BB\uFFFD]")

# ----------------- Helper utilities -----------------
def looks_like_tamil(text):
    """Quick heuristic: return True if string contains Tamil Unicode characters."""
    return bool(RE_TAMIL_CHAR.search(text))

def safe_detect_lang(text):
    if not LANGDETECT_AVAILABLE:
        return None
    try:
        return detect(text)
    except Exception:
        return None

def _clean_latin_input(text):
    """Lowercase and remove stray punctuation that breaks transliteration."""
    t = (text or "").strip()
    # normalize common apostrophe variants
    t = t.replace("’", "'").replace("ʻ", "'")
    # remove punctuation except hyphen & apostrophe to keep tokens readable
    t = re.sub(r"[^\w\s\-']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _tamil_ratio(text):
    """Return fraction of characters that are Tamil script."""
    if not text:
        return 0.0
    total = len(text)
    tamil = len(RE_TAMIL_CHAR.findall(text))
    return tamil / total if total > 0 else 0.0

# ----------------- Cleaning function (Option A fix) -----------------
def clean_transliteration_output(text):
    """
    Clean noisy transliteration outputs:
      - Unicode normalize
      - remove superscripts / stray combining marks
      - remove long ASCII/punctuation blobs
      - collapse whitespace
    Returns cleaned text (may still contain minor issues).
    """
    if not text:
        return text

    # 1) Unicode normalize to avoid odd combining sequences
    t = unicodedata.normalize("NFKC", text)

    # 2) remove common superscript / stray glyphs (these produced the ² ³ artifacts)
    t = _SUPER_SCRIPT_RE.sub(" ", t)

    # 3) remove long ASCII/punctuation runs while preserving Tamil characters and basic spaces
    t = _NON_TAMIL.sub(" ", t)

    # 4) remove combining diacritics (if any)
    t = re.sub(r"[\u0300-\u036F]+", "", t)

    # 5) collapse multiple whitespace into single space
    t = re.sub(r"\s+", " ", t).strip()

    # 6) If cleaning removed nearly all Tamil content, produce a lighter-cleaned fallback
    tamil_chars = len(RE_TAMIL_CHAR.findall(t))
    if tamil_chars < 1:
        # remove only superscripts but keep ascii if needed (safer fallback)
        t2 = _SUPER_SCRIPT_RE.sub("", unicodedata.normalize("NFKC", text))
        t2 = re.sub(r"\s+", " ", t2).strip()
        return t2

    return t

# ----------------- Transliteration helpers -----------------
def transliterate_latn_to_tamil_whole(text):
    """
    Use Aksharamukha to transliterate entire input (Latin->Tamil) if available.
    Validate result by Tamil ratio threshold; return transliteration or None.
    """
    if not AK_AVAILABLE:
        return None

    cleaned = _clean_latin_input(text)
    # try likely schemes; some inputs respond better to different schemes
    schemes = ["ITRANS", "ISO", "HK", "Latin"]
    for src in schemes:
        try:
            out = ak_trans.process(src, "Tamil", cleaned)
            out = re.sub(r"\s+", " ", out).strip()
            if _tamil_ratio(out) >= 0.60:  # require 60% Tamil char coverage
                return out
        except Exception:
            continue

    # final attempt with Latin
    try:
        out = ak_trans.process("Latin", "Tamil", cleaned)
        out = re.sub(r"\s+", " ", out).strip()
        if _tamil_ratio(out) >= 0.60:
            return out
    except Exception:
        pass

    return None

# ----------------- Token-level fallback -----------------
def token_level_normalize(text):
    """
    Per-token fallback: for each whitespace-separated token:
      - if token contains Tamil script, keep
      - if token is Latin-only, try transliterate token (Aksharamukha)
      - else fall back to translator.translate(token)
    Returns joined string.
    """
    if not text:
        return text

    cleaned = _clean_latin_input(text)
    tokens = re.split(r"(\s+)", cleaned)  # keep whitespace separators
    out_tokens = []

    for tok in tokens:
        if tok.strip() == "":
            out_tokens.append(tok)
            continue
        # if token already contains Tamil characters, keep as-is
        if looks_like_tamil(tok):
            out_tokens.append(tok)
            continue
        # latin-only token
        if RE_WORD.fullmatch(tok):
            # try token transliteration (single-token is less likely to produce heavy diacritics)
            translit = None
            if AK_AVAILABLE:
                try:
                    translit = ak_trans.process("Latin", "Tamil", tok)
                    translit = re.sub(r"\s+", " ", translit).strip()
                except Exception:
                    translit = None
            if translit and _tamil_ratio(translit) >= 0.25:
                out_tokens.append(translit)
                continue
            # fallback to translator for the token (if translator available)
            if TRANSLATOR_AVAILABLE and translator is not None:
                try:
                    tr = translator.translate(tok)
                    out_tokens.append(tr)
                    continue
                except Exception:
                    out_tokens.append(tok)
                    continue
            # final fallback: keep original token
            out_tokens.append(tok)
            continue
        # otherwise (punctuation, numbers) keep as-is
        out_tokens.append(tok)

    joined = "".join(out_tokens)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined

# ----------------- Public normalize function (Option A) -----------------
def normalize_to_standard_tamil_improved(text):
    """
    Master pipeline (safer):
      - If input already contains Tamil script -> do light cleaning and return (no transliteration).
      - Else try whole-text transliteration (Aksharamukha) and validate.
      - Else try whole-text translation (deep-translator).
      - Else token-level fallback.
      - Clean final output to remove transliteration artifacts.
    Returns: (final_text, method) where method in {'cleaned-tamil','transliterate','translate','token-level','none'}
    """
    s = (text or "").strip()
    if not s:
        return s, "none"

    # --- 0) If input already has Tamil characters, assume it's Tamil: only clean it and return ---
    if looks_like_tamil(s):
        cleaned = clean_transliteration_output(s)
        # if cleaning removed nearly all Tamil (unlikely), keep original
        if _tamil_ratio(cleaned) < 0.05:
            return s, "cleaned-tamil"
        return cleaned, "cleaned-tamil"

    # --- 1) Whole-text transliteration (only for non-Tamil inputs) ---
    translit_whole = transliterate_latn_to_tamil_whole(s)
    if translit_whole:
        cleaned = clean_transliteration_output(translit_whole)
        # if cleaning removed too many Tamil chars fall back to raw translit_whole
        if _tamil_ratio(cleaned) < 0.10:
            cleaned = translit_whole
        return cleaned, "transliterate"

    # --- 2) Whole-text translation (if translator available) ---
    translated = None
    if TRANSLATOR_AVAILABLE and translator is not None:
        try:
            translated = translator.translate(s)
            if _tamil_ratio(translated) >= 0.25:
                cleaned = clean_transliteration_output(translated)
                return cleaned, "translate"
        except Exception:
            translated = None

    # --- 3) Token-level fallback ---
    tokenized = token_level_normalize(s)
    tokenized_clean = clean_transliteration_output(tokenized)
    if _tamil_ratio(tokenized_clean) >= 0.12:
        return tokenized_clean, "token-level"

    # --- 4) Prefer translator if it existed, else final cleanup of original ---
    if translated:
        return clean_transliteration_output(translated), "translate"
    final = clean_transliteration_output(s)
    return final, "none"

# ----------------- Translator convenience -----------------
def translate_to_tamil(text):
    """Synchronous translator wrapper (deep-translator). Returns original text on failure."""
    if TRANSLATOR_AVAILABLE and translator is not None:
        try:
            return translator.translate(text)
        except Exception:
            return text
    return text

# ----------------- (Optional) small CLI test when run directly -----------------
if __name__ == "__main__":
    examples = [
        "paiththiyam raj odiddan",
        "Karumam Maari irukku Ajith oda dialogue delivery",
        "கருமம் மாரி இருக்கு அஜித்² ஓத³ தி³அலோகு³ஏ தே³லிவேர்ய்",
        "this is an english sentence to translate"
    ]
    for s in examples:
        norm, method = normalize_to_standard_tamil_improved(s)
        print("INPUT :", s)
        print("METHOD:", method)
        print("OUTPUT:", norm)
        print("-" * 40)
