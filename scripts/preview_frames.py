"""Render a contact sheet of authored clips so the geometry can be checked by eye.

Usage: python scripts/preview_frames.py [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vigil.video.scenarios import all_scenarios
from vigil.video.synth import HEIGHT, WIDTH, render_frame

COLS, ROWS, LABEL = 4, 2, 34


def sheet(out: Path) -> Path:
    scenarios = all_scenarios()
    tw, th = WIDTH // 2, HEIGHT // 2
    canvas = Image.new("RGB", (COLS * tw, ROWS * (th + LABEL)), (12, 12, 14))
    draw = ImageDraw.Draw(canvas)
    for i, scenario in enumerate(scenarios):
        moment = scenario.hazards[0].t_impact - 1.2 if scenario.hazards else scenario.duration_s / 2
        frame = render_frame(scenario, moment).resize((tw, th), Image.LANCZOS)
        x, y = (i % COLS) * tw, (i // COLS) * (th + LABEL)
        canvas.paste(frame, (x, y))
        draw.text((x + 4, y + th + 6), f"{scenario.id} @ {moment:.1f}s", fill=(220, 224, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/preview_sheet.png")
    print(sheet(target))
