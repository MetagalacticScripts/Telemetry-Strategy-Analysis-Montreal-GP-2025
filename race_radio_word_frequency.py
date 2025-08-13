import re
import pickle
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
PICKLE_PATH = Path(
    "/Users/audrey/PycharmProjects/Telemetry-Strategy-Analysis-Montreal-GP-2025/"
    "fastf1_cache/2025/2025-06-15_Canadian_Grand_Prix/2025-06-15_Race/"
    "race_control_messages.ff1pkl"
)
TOP_N = 15
OUT_PHRASES = "race_control_top_phrases.png"
OUT_EVENTS  = "race_control_event_counts.png"

# Base stopwords (no NLTK needed)
EN_STOP = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","from","of","in","on","at","by",
    "with","about","as","into","like","through","after","over","between","out","against","during","without","before",
    "under","around","among","this","that","these","those","it","its","they","them","their","there","here","we","us",
    "our","you","your","i","me","my","is","am","are","was","were","be","been","being","do","does","did","doing",
    "have","has","had","having","he","him","his","she","her","hers","who","whom","which","what","why","how","not",
    "no","yes","ok","okay"
}

# Domain stopwords you said were boring + other boilerplate
DOMAIN_STOP = {
    "car","cars","lap","laps","time","timed","track","turn","sector",
    "blue","yellow","green","flag","flags","waved","clear",
    "pit","lane","exit","open","closed",
    "fia","stewards","incident","investigation","noted","reviewed",
    "drs","enabled","disabled","message","driver","drivers","involving"
}

STOPWORDS = EN_STOP  # start conservative; flip to EN_STOP | DOMAIN_STOP if you want only “high-signal” phrases
# STOPWORDS = EN_STOP | DOMAIN_STOP


def load_rc_messages(path: Path):
    """Return a dict-like with 'data' or a DataFrame and also a messages list."""
    try:
        obj = pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    if isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
        msgs = data.get("Message") or data.get("MessageText") or data.get("Text") or []
        msgs = [m for m in msgs if m is not None]
        return data, msgs

    if isinstance(obj, pd.DataFrame):
        for col in ("Message", "MessageText", "Text", "msg"):
            if col in obj.columns:
                return obj, obj[col].astype(str).tolist()
        raise KeyError(f"No text column found. Columns: {list(obj.columns)}")

    raise TypeError(f"Unsupported object type: {type(obj)}")


def extract_driver_codes(messages):
    """Find 3-letter driver codes in patterns like 'CAR 30 (LAW)' and return as lowercase set."""
    codes = set()
    pat = re.compile(r"\((?P<code>[A-Z]{3})\)")
    for m in messages:
        for hit in pat.findall(m):
            codes.add(hit.lower())
    return codes


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split() if len(t) > 1]


def make_bigrams(tokens):
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def plot_barh(labels, values, title, outfile):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values)
    # highlight top
    if bars:
        bars[0].set_color("#FFD700")
        for b in bars[1:]:
            b.set_color("#1f77b4")

    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    # value labels
    for b, v in zip(bars, values):
        ax.text(b.get_width() + max(values)*0.01, b.get_y() + b.get_height()/2, str(v),
                va="center", ha="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved -> {outfile}")
    plt.show()


def main():
    data_obj, messages = load_rc_messages(PICKLE_PATH)

    # Auto-stopwords: driver codes eg. (LAW), (BEA), etc.
    auto_codes = extract_driver_codes(messages)  # lowercased
    dynamic_stop = STOPWORDS | auto_codes

    # Tokenize and filter
    tokens_raw = []
    for m in messages:
        tokens_raw.extend(tokenize(m))

    tokens = [t for t in tokens_raw if t not in dynamic_stop]

    # Build bigrams and filter ones where both tokens are stopwords
    bigrams = make_bigrams(tokens)

    # If you want even “cleaner” phrases, you can also drop bigrams that contain any stopword:
    # bigrams = [bg for bg in bigrams if all(w not in dynamic_stop for w in bg.split())]

    counts = Counter(bigrams).most_common(TOP_N)
    if counts:
        labels, values = zip(*counts)
        plot_barh(labels, values, f"Top {TOP_N} Phrases – Race Control Messages (Montreal GP 2025)", OUT_PHRASES)
    else:
        print("No bigrams left after filtering; consider loosening STOPWORDS or keeping DOMAIN_STOP off.")

    # ----- Event mix from structured fields -----
    # Build a DataFrame view from dict 'data' if needed
    if isinstance(data_obj, dict) and "Time" in data_obj:
        df = pd.DataFrame(data_obj)
    elif isinstance(data_obj, pd.DataFrame):
        df = data_obj.copy()
    else:
        df = None

    if df is not None:
        # Category counts
        cat = df["Category"].dropna().astype(str).value_counts().head(10)
        # Flag counts
        flg = df["Flag"].dropna().astype(str).value_counts().head(10)

        # Combine for a single chart (stacked) or make two charts; we’ll do two small stacked columns:
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        cat.plot(kind="barh", ax=axes[0], color="#1f77b4")
        axes[0].invert_yaxis()
        axes[0].set_title("Category Counts", fontsize=12, weight="bold", pad=6)
        axes[0].grid(axis="x", linestyle="--", alpha=0.35)

        flg.plot(kind="barh", ax=axes[1], color="#2ca02c")
        axes[1].invert_yaxis()
        axes[1].set_title("Flag Counts", fontsize=12, weight="bold", pad=6)
        axes[1].grid(axis="x", linestyle="--", alpha=0.35)

        fig.suptitle("Race Control – Event Mix (Montreal GP 2025)", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(OUT_EVENTS, dpi=300, bbox_inches="tight")
        print(f"Saved -> {OUT_EVENTS}")
        plt.show()
    else:
        print("Could not construct DataFrame for event counts.")


if __name__ == "__main__":
    main()
