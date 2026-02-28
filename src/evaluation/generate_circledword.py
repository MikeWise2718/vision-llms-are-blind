"""Generate CircledWord benchmark images with ground truth encoded in filenames.

Adapted from src/CircledWord/text_image_generator.py + GenerateSamples.ipynb.
Each image shows a word with a red ellipse circling one letter.

Filename format:
    circled_{word}_idx{N}_char{C}_t{thickness}_p{padding}_f{font_id}.png

    idx  = 0-based index of the circled letter
    char = the actual circled letter (ground truth answer)
    t    = ellipse line thickness
    p    = horizontal padding around the word
    f    = font index (0-based)

Parameters match the original study:
    - 3 words: Acknowledgement (15 chars), Subdermatoglyphic (17 chars), tHyUiKaRbNqWeOpXcZvM (20 chars)
    - 2 fonts
    - thickness: 4, 5, 6
    - padding: 25, 50, 100, 200
    - scale_factor: 1.4
    - final size: 512x512

Expected output: 15*2*3*4 + 17*2*3*4 + 20*2*3*4 = 360 + 408 + 480 = 1,248 images

Usage:
    python -m src.evaluation.generate_circledword [--output-dir DIR] [--fonts-dir DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import freetype
import numpy as np
from PIL import Image


# Generation parameters (matching original study)
WORDS = ["Acknowledgement", "Subdermatoglyphic", "tHyUiKaRbNqWeOpXcZvM"]
THICKNESSES = [4, 5, 6]
PADDINGS = [25, 50, 100, 200]
SCALE_FACTOR = 1.4
CANVAS_WIDTH = 10   # inches
CANVAS_HEIGHT = 2   # inches
FINAL_WIDTH = 512   # pixels
FINAL_HEIGHT = 512  # pixels
PIXEL_HEIGHT = 96
CHAR_SIZE = 36 * 64


def load_font(font_path):
    """Load a font face with freetype."""
    face = freetype.Face(font_path)
    face.set_char_size(CHAR_SIZE)
    face.set_pixel_sizes(0, PIXEL_HEIGHT)
    return face


def draw_glyph(ax, face, char, position):
    """Render a single glyph onto the axes. Returns advance position and glyph center."""
    face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    bitmap = face.glyph.bitmap
    top = face.glyph.bitmap_top
    left = face.glyph.bitmap_left

    x = position[0] + left
    y = 200 - (position[1] - top)

    buffer_array = np.array(bitmap.buffer, dtype=np.uint8)
    if bitmap.width > 0 and bitmap.rows > 0:
        buffer_reshaped = buffer_array.reshape(bitmap.rows, bitmap.width)
        buffer_inverted = 255 - buffer_reshaped
        ax.imshow(
            buffer_inverted,
            cmap="gray",
            interpolation="bilinear",
            extent=(x, x + bitmap.width, y - bitmap.rows, y),
        )

    center_x = x + bitmap.width / 2
    center_y = y - bitmap.rows / 2
    return position[0] + face.glyph.advance.x // 64, position[1], center_x, center_y


def create_image(
    text, font_path, circle_index, thickness, scale_factor, padding,
    canvas_width, canvas_height, final_width, final_height, output_path, tmp_dir,
):
    """Generate a single CircledWord image and save to output_path."""
    face = load_font(font_path)

    # Measure glyphs
    total_width = 0
    max_height = 0
    glyph_sizes = []
    for char in text:
        face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        glyph_width = face.glyph.advance.x // 64
        glyph_height = face.glyph.bitmap_top + face.glyph.bitmap.rows - face.glyph.bitmap_top
        glyph_sizes.append((glyph_width, glyph_height))
        total_width += glyph_width
        if glyph_height > max_height:
            max_height = glyph_height

    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height), dpi=1200)
    ax.set_frame_on(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlim(0, total_width + 2 * padding)
    ax.set_ylim(0, 200)

    vertical_center = 100 + (max_height // 2)
    x, y = (padding, vertical_center)
    centers = []
    for i, (char, (glyph_width, glyph_height)) in enumerate(zip(text, glyph_sizes)):
        x, y, center_x, center_y = draw_glyph(ax, face, char, (x, y))
        centers.append((center_x, center_y))
        if i == circle_index:
            radius_x = (glyph_width / 2) * scale_factor
            radius_y = (glyph_height / 2) * scale_factor
            if 0 <= i < len(centers):
                cx, cy = centers[i]
                ellipse = patches.Ellipse(
                    (cx, cy),
                    width=2 * radius_x,
                    height=2 * radius_y,
                    fill=False,
                    edgecolor="red",
                    linewidth=thickness,
                )
                ax.add_patch(ellipse)

    ax.set_xticks([])
    ax.set_yticks([])

    # Save high-res temporary image
    tmp_path = os.path.join(tmp_dir, f"tmp_{os.getpid()}_{circle_index}.png")
    fig.savefig(tmp_path, dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Resize to final dimensions maintaining aspect ratio, centered on white background
    image = Image.open(tmp_path)
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if final_width / final_height > aspect_ratio:
        new_height = final_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = final_width
        new_height = int(new_width / aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    final_image = Image.new("RGB", (final_width, final_height), "white")
    paste_x = (final_width - new_width) // 2
    paste_y = (final_height - new_height) // 2
    final_image.paste(resized_image, (paste_x, paste_y))
    final_image.save(output_path)

    # Clean up temp
    os.remove(tmp_path)
    return output_path


def get_fonts(fonts_dir):
    """Get list of font files in the fonts directory."""
    if not os.path.isdir(fonts_dir):
        return []
    fonts = []
    for f in sorted(os.listdir(fonts_dir)):
        if f.lower().endswith((".ttf", ".otf")):
            fonts.append(os.path.join(fonts_dir, f))
    return fonts


def generate_all(output_dir, fonts_dir):
    """Generate the full CircledWord image set."""
    fonts = get_fonts(fonts_dir)
    if not fonts:
        print(f"No fonts found in {fonts_dir}")
        print("Run: python -m src.evaluation.setup_fonts")
        return 0

    print(f"Found {len(fonts)} font(s):")
    for i, f in enumerate(fonts):
        print(f"  [{i}] {os.path.basename(f)}")

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    total = 0
    expected = sum(len(w) * len(fonts) * len(THICKNESSES) * len(PADDINGS) for w in WORDS)
    print(f"\nGenerating {expected} images...")

    for word in WORDS:
        for font_idx, font_path in enumerate(fonts):
            for circle_index in range(len(word)):
                char = word[circle_index]
                for thickness in THICKNESSES:
                    for padding in PADDINGS:
                        filename = (
                            f"circled_{word}_idx{circle_index}_char{char}"
                            f"_t{thickness}_p{padding}_f{font_idx}.png"
                        )
                        output_path = os.path.join(output_dir, filename)

                        if os.path.exists(output_path):
                            total += 1
                            continue

                        try:
                            create_image(
                                text=word,
                                font_path=font_path,
                                circle_index=circle_index,
                                thickness=thickness,
                                scale_factor=SCALE_FACTOR,
                                padding=padding,
                                canvas_width=CANVAS_WIDTH,
                                canvas_height=CANVAS_HEIGHT,
                                final_width=FINAL_WIDTH,
                                final_height=FINAL_HEIGHT,
                                output_path=output_path,
                                tmp_dir=tmp_dir,
                            )
                            total += 1
                        except Exception as e:
                            print(f"  Error: {filename}: {e}")

                        if total % 100 == 0:
                            print(f"  {total}/{expected} images generated...")

    # Clean up tmp dir
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\nDone: {total} images in {output_dir}")
    return total


def main():
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "generated_images", "CircledWord"
    )
    default_fonts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

    parser = argparse.ArgumentParser(description="Generate CircledWord benchmark images")
    parser.add_argument("-o", "--output-dir", default=default_output, help="Output directory")
    parser.add_argument("-f", "--fonts-dir", default=default_fonts, help="Fonts directory")
    args = parser.parse_args()

    generate_all(args.output_dir, args.fonts_dir)


if __name__ == "__main__":
    main()
