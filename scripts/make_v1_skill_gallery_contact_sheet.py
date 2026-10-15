from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v1_skill_figure_gallery_20260816"


ITEMS = [
    ("scientific-visualization", OUT / "fig01_scientific_visualization.png"),
    ("nature-figure", OUT / "fig02_nature_figure.png"),
    ("paper-figure", OUT / "fig03_paper_figure.png"),
    ("academic-plotting", OUT / "fig04_academic_plotting.png"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (title, path) in zip(axes.ravel(), ITEMS):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93, wspace=0.05, hspace=0.12)
    fig.savefig(OUT / "contact_sheet_skill_versions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
