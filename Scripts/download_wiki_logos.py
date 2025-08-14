# download_wiki_logos.py  (patched for Wikipedia 403)
from pathlib import Path
import re
import io
import time
import requests
from bs4 import BeautifulSoup
from PIL import Image

try:
    import cairosvg
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

LOGO_DIR = Path("logos")
LOGO_DIR.mkdir(parents=True, exist_ok=True)

TEAM_PAGES = {
    "Mercedes": "Mercedes-AMG Petronas F1 Team",
    "Ferrari": "Scuderia Ferrari",
    "Red Bull": "Red Bull Racing",
    "McLaren": "McLaren",
    "Aston Martin": "Aston Martin Aramco F1 Team",
    "Alpine": "BWT Alpine F1 Team",
    "Williams": "Williams Grand Prix Engineering",
    "RB (VCARB)": "Visa Cash App RB F1 Team",
    "Sauber": "Stake F1 Team Kick Sauber",
    "Haas": "Haas F1 Team",
}

WIKI_BASE = "https://en.wikipedia.org/wiki/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    )
}

def pick_infobox_logo(img_tags):
    def score(img):
        alt = (img.get("alt") or "").lower()
        src = (img.get("src") or "").lower()
        s = 0
        if "logo" in alt or "logo" in src:
            s += 10
        if "car" in alt:
            s -= 3
        if ".svg" in src:
            s += 2
        return s

    if not img_tags:
        return None
    return sorted(img_tags, key=score, reverse=True)[0]

def to_absolute_url(src: str) -> str:
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return "https://en.wikipedia.org" + src
    return src

def fetch_infobox_logo_url(page_title: str) -> str | None:
    url = WIKI_BASE + page_title.replace(" ", "_")
    r = requests.get(url, timeout=15, headers=HEADERS)
    if not r.ok:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    infobox = soup.find("table", class_=re.compile(r"\binfobox\b"))
    if not infobox:
        return None
    imgs = infobox.find_all("img")
    if not imgs:
        return None
    img = pick_infobox_logo(imgs)
    if not img:
        return None
    return to_absolute_url(img.get("src", ""))

def save_png_from_response(content: bytes, dest_png: Path):
    with Image.open(io.BytesIO(content)) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        im.save(dest_png, format="PNG")

def download_and_convert(url: str, dest_png: Path):
    r = requests.get(url, timeout=20, headers=HEADERS)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}")

    ct = r.headers.get("Content-Type", "").lower()
    if ct.endswith("svg+xml") or url.lower().endswith(".svg"):
        if not HAS_CAIROSVG:
            raise RuntimeError("SVG image but cairosvg not installed.")
        cairosvg.svg2png(bytestring=r.content, write_to=dest_png.as_posix())
    else:
        save_png_from_response(r.content, dest_png)

def main():
    for team, page in TEAM_PAGES.items():
        dest = LOGO_DIR / f"{team}.png"
        try:
            print(f"[{team}] fetching infobox logo …")
            img_url = fetch_infobox_logo_url(page)
            if not img_url:
                print(f"  ! No image found on page: {page}")
                continue
            img_url = re.sub(r"/(\d{2,4})px-", "/400px-", img_url)
            download_and_convert(img_url, dest)
            print(f"  -> saved {dest}")
        except Exception as e:
            print(f"  ! Failed for {team}: {e}")
        time.sleep(1.5)  # pause between requests

    print("\nDone. Check the logos/ directory.")

if __name__ == "__main__":
    main()
