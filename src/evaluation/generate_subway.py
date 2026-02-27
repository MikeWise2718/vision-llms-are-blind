"""Generate SubwayMap benchmark images with ground truth encoded in filenames.

Adapted from src/SubwayMap/SubwayMap.ipynb. Each image contains 4 stations
(A, B, C, D) connected by single-colored paths on an 18x18 grid.

Filename format:
    subway_s{size}_lw{thickness}_AB{n}_AC{n}_AD{n}_BC{n}_BD{n}_CD{n}_seed{S}.png

This encodes the path count for every station pair, enabling ground truth
extraction without external metadata files.

Usage:
    python -m src.evaluation.generate_subway [--output-dir DIR] [--seeds N]
"""

import argparse
import copy
import os
import random
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import numpy as np
from PIL import Image


# Grid constants (from original notebook)
VALID_MOVES = {
    "A": [(0, -1), (-1, 0), (1, 0)],
    "B": [(0, -1), (0, 1), (-1, 0)],
    "C": [(0, 1), (-1, 0), (1, 0)],
    "D": [(0, -1), (0, 1), (1, 0)],
}

STARTING_MOVES = {
    "A": (0, -1),
    "B": (-1, 0),
    "C": (0, 1),
    "D": (1, 0),
}

STATIONS_POINTS = {
    "A": [(8, 16), (9, 16), (10, 16)],
    "B": [(16, 8), (16, 9), (16, 10)],
    "C": [(8, 2), (9, 2), (10, 2)],
    "D": [(2, 8), (2, 9), (2, 10)],
}

NODE_TO_LABEL = {}
for k, v in STATIONS_POINTS.items():
    for pt in v:
        NODE_TO_LABEL[str(pt)] = k

STATIONS = ["A", "B", "C", "D"]
STATION_PAIRS = list(combinations(STATIONS, 2))  # AB, AC, AD, BC, BD, CD


def get_colors_from_colormap(colormap_name: str, num_colors: int) -> list:
    colormap = colormaps[colormap_name]
    return [colormap(i) for i in np.arange(num_colors)]


def rgba_to_color_name(rgba) -> str:
    colors = mcolors.CSS4_COLORS
    input_rgb = rgba[:3]
    return min(colors, key=lambda name: np.linalg.norm(
        np.array(mcolors.to_rgba(colors[name])[:3]) - np.array(input_rgb)
    ))


def generate_routes(path_count: int, rng: random.Random) -> list[dict] | None:
    """Generate a set of routes connecting stations, each station having exactly path_count connections.

    Returns list of route dicts or None if generation fails.
    """
    stations = list(STATIONS)
    rng.shuffle(stations)

    path_counter = defaultdict(int)
    visited = []
    all_routes = []
    cross_dest = []

    for station in stations:
        path_counter[station] += 0

    for station in stations:
        start_list = copy.deepcopy(STATIONS_POINTS[station])
        rng.shuffle(start_list)

        possible_destinations = []
        for k, v in STATIONS_POINTS.items():
            if k == station:
                continue
            if path_counter[k] == path_count:
                continue
            for pt in v:
                if pt not in cross_dest:
                    possible_destinations.append(pt)

        for _ in range(1, path_count + 1):
            if path_counter[station] == path_count:
                break

            start = start_list.pop()
            root = start
            end = (start[0] + STARTING_MOVES[station][0],
                   start[1] + STARTING_MOVES[station][1])

            routes = {"path": []}

            while True:
                if end in possible_destinations:
                    break

                if ([start, end] not in visited and [end, start] not in visited
                        and 2 < end[0] < 16 and 2 < end[1] < 16):
                    routes["path"].append([start, end])
                    visited.append([start, end])
                    temp_moves = copy.deepcopy(VALID_MOVES[station])
                else:
                    if len(routes["path"]) < 1:
                        break
                    old_data = routes["path"][-1]
                    start, end = old_data[0], old_data[1]

                if len(temp_moves) < 1:
                    if len(routes["path"]) > 1:
                        routes["path"].pop(-1)
                    else:
                        break
                    old_data = routes["path"][-1]
                    start, end = old_data[0], old_data[1]
                    temp_moves = copy.deepcopy(VALID_MOVES[station])

                route = rng.choice(temp_moves)
                temp_moves.remove(route)
                start = end
                end = (start[0] + route[0], start[1] + route[1])

            if (end in possible_destinations
                    and [start, end] not in visited
                    and [end, start] not in visited):
                routes["path"].append([start, end])
                visited.append([start, end])
                all_routes.append(routes)
                path_counter[station] += 1
                cross_dest.append(end)
                cross_dest.append(root)

                for k, v in STATIONS_POINTS.items():
                    if end in v:
                        path_counter[k] += 1

    # Check all stations have exactly path_count connections
    if all(path_counter[s] == path_count for s in stations):
        return all_routes
    return None


def count_pair_paths(all_routes: list[dict]) -> dict[str, int]:
    """Count paths between each station pair from route data."""
    pair_counts = {f"{a}{b}": 0 for a, b in STATION_PAIRS}

    for route in all_routes:
        start_label = NODE_TO_LABEL[str(route["path"][0][0])]
        end_label = NODE_TO_LABEL[str(route["path"][-1][-1])]
        pair = "".join(sorted([start_label, end_label]))
        pair_counts[pair] += 1

    return pair_counts


def draw_subway(all_routes: list[dict], thickness: int, image_size: int) -> Image.Image:
    """Render routes to a PIL image."""
    colors = get_colors_from_colormap("tab10", len(all_routes))

    fig, ax = plt.subplots(figsize=(18, 18), dpi=500)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.axis("off")

    ax.text(8.45, 16.5, "A", fontsize=100, color="k", fontweight="bold")
    ax.text(16.5, 8.6, "B", fontsize=100, color="k", fontweight="bold")
    ax.text(8.45, 0.6, "C", fontsize=100, color="k", fontweight="bold")
    ax.text(0.4, 8.6, "D", fontsize=100, color="k", fontweight="bold")

    for i, route in enumerate(all_routes):
        for segment in route["path"]:
            x1, y1 = segment[0]
            x2, y2 = segment[1]
            ax.plot([x1, x2], [y1, y2], color=colors[i], linestyle="solid", linewidth=thickness)

    plt.tight_layout(pad=0.0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(buf[..., :3]).resize((image_size, image_size))
    plt.close(fig)
    return image


def generate_images(output_dir: str, num_seeds: int = 15):
    """Generate subway map images with ground truth in filenames.

    For each seed, tries path counts 1, 2, 3. For each successful generation,
    renders at 2 sizes × 2 thicknesses = 4 images.
    """
    os.makedirs(output_dir, exist_ok=True)

    path_counts = [1, 2, 3]
    image_sizes = [512, 1024]
    thicknesses = [10, 20]

    total = 0
    for seed in range(num_seeds):
        for pc in path_counts:
            rng = random.Random(seed * 100 + pc)
            # Try generating routes (may need multiple attempts with different sub-seeds)
            routes = None
            for attempt in range(50):
                rng_attempt = random.Random(seed * 10000 + pc * 100 + attempt)
                routes = generate_routes(pc, rng_attempt)
                if routes is not None:
                    break

            if routes is None:
                print(f"  Seed {seed}, paths={pc}: failed to generate (skipped)")
                continue

            pair_counts = count_pair_paths(routes)

            for size in image_sizes:
                for thickness in thicknesses:
                    # Build filename with all pair counts
                    pairs_str = "_".join(f"{k}_{v}" for k, v in sorted(pair_counts.items()))
                    fname = f"subway_s{size}_lw{thickness}_{pairs_str}_seed{seed}p{pc}.png"
                    fpath = os.path.join(output_dir, fname)

                    image = draw_subway(routes, thickness, size)
                    image.save(fpath)
                    total += 1

    print(f"Generated {total} subway map images in {output_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate SubwayMap benchmark images")
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "SubwayMap"),
        help="Output directory for generated images",
    )
    parser.add_argument(
        "-s", "--seeds", type=int, default=15,
        help="Number of random seeds to use (default: 15)",
    )
    args = parser.parse_args()
    generate_images(args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
