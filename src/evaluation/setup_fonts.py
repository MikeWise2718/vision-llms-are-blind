"""Download Google Fonts for CircledWord image generation.

Downloads Roboto Regular and Open Sans Regular TTF files to src/evaluation/fonts/.
These are freely-licensed fonts used to generate CircledWord benchmark images.

Usage:
    python -m src.evaluation.setup_fonts
"""

import os
import urllib.request

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Google Fonts direct download URLs (static TTF files from github.com/google/fonts)
FONTS = {
    "Roboto-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
        # Variable font — we download and rename; freetype handles it fine
        "Roboto-Regular.ttf",
    ),
    "OpenSans-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
        "OpenSans-Regular.ttf",
    ),
}

# Fallback: static TTF URLs from older Google Fonts repo snapshots
FONTS_FALLBACK = {
    "Roboto-Regular.ttf": "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-android.zip",
    "OpenSans-Regular.ttf": "https://fonts.google.com/download?family=Open+Sans",
}


def download_fonts(fonts_dir: str = FONTS_DIR) -> list[str]:
    """Download fonts to the specified directory. Returns list of downloaded paths."""
    os.makedirs(fonts_dir, exist_ok=True)

    downloaded = []
    for local_name, (url, filename) in FONTS.items():
        dest = os.path.join(fonts_dir, filename)
        if os.path.exists(dest):
            print(f"  Already exists: {filename}")
            downloaded.append(dest)
            continue

        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
            downloaded.append(dest)
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            print(f"  You can manually download fonts to {fonts_dir}")

    return downloaded


def main():
    print(f"Downloading fonts to {FONTS_DIR}")
    paths = download_fonts()
    print(f"\n{len(paths)} font(s) ready in {FONTS_DIR}")
    if not paths:
        print("No fonts downloaded. Please manually place TTF/OTF files in the fonts directory.")
    return paths


if __name__ == "__main__":
    main()
