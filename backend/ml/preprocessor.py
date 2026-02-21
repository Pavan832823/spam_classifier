"""
Text Preprocessing Pipeline for Email Spam Classifier
PRODUCTION-READY: All bugs fixed.

FIXED:
  1. Stemmer had 9 duplicate suffix rules (dead code / non-deterministic order).
     Deduplicated; kept the most useful variant of each pair.
"""

import re
import string

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both",
    "each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","s","t","can",
    "will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn",
}


class TextPreprocessor:
    """
    Cleans and normalises raw email text for ML feature extraction.

    Pipeline:
      1. Lowercase
      2. Strip HTML tags
      3. Replace URLs  → TOKEN_URL
      4. Replace emails → TOKEN_EMAIL
      5. Replace currency → TOKEN_MONEY
      6. Replace numbers → TOKEN_NUMBER
      7. Remove punctuation (preserving underscores for tokens)
      8. Tokenise
      9. Remove stopwords
      10. Suffix stemming
    """

    # BUG FIXED: removed 9 duplicate suffix rules from original.
    # Kept the most semantically useful variant for each duplicated pair.
    _SUFFIXES = [
        ("ational", "ate"),  ("tional",  "tion"), ("enci",   "ence"),
        ("anci",    "ance"), ("izing",   "ize"),   ("ising",  "ise"),
        ("nesses",  ""),     ("fulness", ""),      ("iveness",""),
        ("alities", "al"),   ("ations",  "ate"),   ("ation",  "ate"),
        ("ements",  ""),     ("ement",   ""),      ("ments",  ""),
        ("ment",    ""),     ("ings",    ""),      ("ing",    ""),
        ("edly",    ""),     ("ingly",   ""),      ("iness",  "y"),
        ("ness",    ""),     ("fully",   ""),      ("ably",   ""),
        ("ibly",    ""),     ("ised",    "ise"),   ("ized",   "ize"),
        ("ities",   "ity"),  ("ity",     ""),      ("ives",   "ive"),
        ("ive",     ""),     ("ises",    "ise"),   ("izes",   "ize"),
        ("ated",    "ate"),  ("ates",    "ate"),   ("ate",    ""),
        ("iers",    "y"),    ("iest",    "y"),      ("est",   ""),
        ("ier",     "y"),    ("ies",     "y"),      ("ied",   "y"),
        ("tion",    "te"),   ("sion",    "d"),      ("ally",  "al"),
        ("ily",     "y"),    ("ly",      ""),       ("ful",   ""),
        ("al",      ""),     ("ers",     ""),       ("er",    ""),
        ("ous",     ""),     ("ism",     ""),       ("ist",   ""),
        ("ed",      ""),     ("es",      ""),       ("s",     ""),
    ]

    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming      = use_stemming
        self.remove_stopwords  = remove_stopwords
        # Pre-compile translation table once
        punct_no_underscore = string.punctuation.replace("_", "")
        self._punct_table   = str.maketrans(
            punct_no_underscore, " " * len(punct_no_underscore)
        )

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self._strip_html(text)
        text = self._replace_urls(text)
        text = self._replace_emails(text)
        text = self._replace_currency(text)
        text = self._replace_numbers(text)
        text = text.translate(self._punct_table)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        if self.use_stemming:
            tokens = [self._stem(t) for t in tokens]
        return " ".join(tokens)

    def _strip_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _replace_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " TOKEN_URL ", text)

    def _replace_emails(self, text: str) -> str:
        return re.sub(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            " TOKEN_EMAIL ", text,
        )

    def _replace_currency(self, text: str) -> str:
        return re.sub(
            r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*\s*(?:usd|dollars?)\b",
            " TOKEN_MONEY ", text, flags=re.IGNORECASE,
        )

    def _replace_numbers(self, text: str) -> str:
        return re.sub(r"\b\d+\b", " TOKEN_NUMBER ", text)

    def _stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        for suffix, replacement in self._SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)] + replacement
        return word


def preprocess_text(text: str) -> str:
    return TextPreprocessor().clean(text)


if __name__ == "__main__":
    p = TextPreprocessor()
    samples = [
        "CONGRATULATIONS! You WON $1,000,000!!! Click http://scam-link.xyz NOW!!! FREE OFFER!!!",
        "Hi, the meeting is scheduled for 3pm. Could you review the attached report? Thanks.",
        "<html><body>Your account at user@bank.com has been <b>SUSPENDED</b>!</body></html>",
    ]
    for text in samples:
        cleaned = p.clean(text)
        print(f"Original: {text[:60]}...")
        print(f"Cleaned:  {cleaned}\n")
