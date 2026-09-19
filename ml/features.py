"""Shared URL feature extraction.

Imported by both the offline training pipeline (`ml/train.py`) and the
live backend (`backend/app/ml_service`) so the exact same features are
computed at training time and at inference time — the model never sees a
feature at serving time that it wasn't trained on, or vice versa.

All features are simple lexical/domain/statistical properties of the URL
string itself (no network calls, no external lookups) — that's what keeps
inference fast and the whole pipeline easy to reason about.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# Words phishing pages commonly use to look legitimate/urgent.
SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "confirm",
    "signin",
    "bank",
    "password",
    "billing",
    "suspend",
]

# A handful of well-known URL shorteners; phishers use these to hide the
# real destination domain.
URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "cutt.ly",
    "rebrand.ly",
}

IP_ADDRESS_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# High-value brands phishing pages most commonly impersonate. Used for
# typosquat detection (brand_edit_distance below): a domain that's a small
# edit away from one of these but not an exact match ("paypa1", "arnazon")
# is exactly the shape of a squat, a signal the raw lexical features above
# can't see on their own.
POPULAR_BRANDS = [
    "google", "facebook", "amazon", "apple", "microsoft", "paypal", "netflix",
    "instagram", "twitter", "linkedin", "ebay", "yahoo", "bankofamerica",
    "wellsfargo", "chase", "citibank", "americanexpress", "dropbox", "adobe",
    "spotify", "walmart", "target", "coinbase", "binance", "steam", "irs",
    "usps", "fedex", "ups", "dhl", "office365", "outlook", "icloud",
    "whatsapp", "telegram", "discord", "github", "gitlab", "youtube",
    "wellsfargo", "hsbc", "barclays", "docusign", "zoom", "twitch",
]  # fmt: skip
_BRAND_EDIT_DISTANCE_CAP = 10

# TLDs long favored by phishing campaigns for free or near-free
# registration -- a signal independent of URL shape/length.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "support",
    "click", "loan", "win", "download", "review", "party", "gdn", "cam",
}  # fmt: skip


def _levenshtein(a: str, b: str) -> int:
    """Iterative edit distance, no extra dependency for ~40 short brand strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _shannon_entropy(s: str) -> float:
    """Bits of entropy per character -- random/DGA-style hostnames score
    high, real words and brand names score low."""
    if not s:
        return 0.0
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length) for count in Counter(s).values()
    )


def _longest_digit_run(s: str) -> int:
    return max((len(run) for run in re.findall(r"\d+", s)), default=0)

# Fixed order so training and serving always build the feature vector the
# same way.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "has_ip_address",
    "has_at_symbol",
    "is_https",
    "digit_ratio",
    "num_query_params",
    "has_suspicious_word",
    "is_url_shortener",
    "hostname_entropy",
    "brand_edit_distance",
    "is_suspicious_tld",
    "is_punycode",
    "hostname_digit_ratio",
    "longest_digit_run",
]


def extract_features(url: str) -> dict:
    """Compute the feature dict for a single URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    ext = tldextract.extract(url)

    num_letters = sum(c.isalpha() for c in url)
    num_digits = sum(c.isdigit() for c in url)

    # Split on "-"/"_" and compare each token, not just the whole label --
    # "paypal-secure.tk" and "paypa1-login.tk" are typosquats hiding a
    # near-exact brand token behind an extra word, which whole-label edit
    # distance (vs. "paypal") would miss by counting the unrelated word's
    # length as divergence too.
    sld_tokens = [t for t in re.split(r"[-_]", ext.domain.lower()) if t]
    brand_edit_distance = (
        min(
            min(_levenshtein(token, brand), _BRAND_EDIT_DISTANCE_CAP)
            for token in sld_tokens
            for brand in POPULAR_BRANDS
        )
        if sld_tokens
        else _BRAND_EDIT_DISTANCE_CAP
    )
    hostname_digits = sum(c.isdigit() for c in hostname)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": num_digits,
        "num_special_chars": sum(not c.isalnum() for c in url),
        "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "has_ip_address": int(bool(IP_ADDRESS_RE.match(hostname))),
        "has_at_symbol": int("@" in url),
        "is_https": int(parsed.scheme == "https"),
        "digit_ratio": num_digits / num_letters if num_letters else 0.0,
        "num_query_params": parsed.query.count("&") + 1 if parsed.query else 0,
        "has_suspicious_word": int(any(word in url.lower() for word in SUSPICIOUS_WORDS)),
        "is_url_shortener": int(ext.top_domain_under_public_suffix in URL_SHORTENERS),
        "hostname_entropy": _shannon_entropy(hostname),
        "brand_edit_distance": brand_edit_distance,
        "is_suspicious_tld": int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS if ext.suffix else 0),
        "is_punycode": int("xn--" in hostname),
        "hostname_digit_ratio": hostname_digits / len(hostname) if hostname else 0.0,
        "longest_digit_run": _longest_digit_run(hostname),
    }


def build_feature_matrix(urls):
    """Vectorized helper for the training pipeline: list/Series of URLs -> DataFrame.

    Only used offline by `ml/train.py` — kept out of the hot inference path
    and out of the backend's dependencies (pandas isn't a serving dependency).
    """
    import pandas as pd

    rows = [extract_features(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
