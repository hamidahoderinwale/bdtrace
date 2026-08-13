"""Generate controlled_eval_temporal_altair.png with a fully visible legend.

Root cause of previous legend clipping:
- Vega-Lite suppresses the legend on line layers when an area (y2) layer is also present
  in the same layered chart. Workaround: put the legend on the AREA (y2) layer instead.
- Save via vl_convert.vegalite_to_png (not Altair .save()) to control rendering directly.
- Verify legend visibility using vision-LLM (OpenRouter anthropic/claude-sonnet-4-6).
"""

import json
import os
import sys
import base64

import altair as alt
import pandas as pd
import vl_convert as vlc

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces"
DATA_PATH = os.path.join(BASE_DIR, "output/paper2_pilot/controlled_eval_temporal.json")
OUT_PATH  = os.path.join(BASE_DIR, "output/figures/controlled_eval_temporal_altair.png")

sys.path.insert(0, BASE_DIR)
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER

register()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# ---------------------------------------------------------------------------
# Add Chance line data (AUC = 0.5 across all k values)
# ---------------------------------------------------------------------------
k_vals = sorted(df["k"].unique())
chance_rows = [
    {"agent": "Chance (AUC = 0.5)", "k": k, "auc_mean": 0.5, "auc_lo": 0.5, "auc_hi": 0.5}
    for k in k_vals
]
df_full = pd.concat([df, pd.DataFrame(chance_rows)], ignore_index=True)

# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------
all_agents_ordered = AGENT_ORDER + ["Chance (AUC = 0.5)"]
color_map = dict(AGENT_COLORS)
color_map["Chance (AUC = 0.5)"] = "#999999"

color_domain = all_agents_ordered
color_range  = [color_map[a] for a in color_domain]

# Theme config (from scripts/theme.py register())
_FONT = "Inter, -apple-system, system-ui, sans-serif"
_THEME_CONFIG = {
    "background": "#FFFFFF",
    "view": {"stroke": None},
    "axis": {
        "labelFont": _FONT, "titleFont": _FONT,
        "labelFontSize": 11, "titleFontSize": 12,
        "labelFontWeight": "normal", "titleFontWeight": "normal",
        "labelColor": "#444444", "titleColor": "#222222",
        "grid": False, "domain": False, "ticks": False,
    },
    "axisX": {"labelAngle": 0},
    "axisY": {"labelLimit": 380},
    "legend": {
        "labelFont": _FONT, "titleFont": _FONT,
        "labelFontSize": 11, "titleFontSize": 11,
        "symbolSize": 80, "padding": 8,
    },
    "title": {
        "font": _FONT, "fontSize": 13, "fontWeight": "normal",
        "color": "#111111", "anchor": "start", "offset": 10,
    },
    "mark": {"font": _FONT},
    "point": {"size": 60, "strokeWidth": 1.5},
    "line": {"strokeWidth": 2},
}

# ---------------------------------------------------------------------------
# Build Vega-Lite spec dict
# ---------------------------------------------------------------------------
CHART_WIDTH  = 500
CHART_HEIGHT = 320


def build_spec(legend_x: int = 520, right_padding: int = 250) -> dict:
    """
    Build a Vega-Lite spec with a visible legend.

    Key insight: in a layered chart with an area (y2) layer, the legend
    must be placed on the AREA layer — placing it on a line layer causes
    Vega-Lite to suppress it silently.
    """
    df_agents = df_full[df_full["agent"] != "Chance (AUC = 0.5)"].copy()
    df_chance = df_full[df_full["agent"] == "Chance (AUC = 0.5)"].copy()

    df_agents_records = df_agents.to_dict(orient="records")
    df_chance_records = df_chance.to_dict(orient="records")

    shared_color_scale = {"domain": color_domain, "range": color_range}

    # Legend spec: orient=none + explicit x/y so it renders outside the plot area
    legend_spec = {
        "title": None,
        "orient": "none",
        "legendX": legend_x,
        "legendY": 10,
        "labelLimit": 220,
        "symbolStrokeWidth": 2,
        "symbolType": "stroke",
    }

    def color_with_legend():
        return {
            "field": "agent", "type": "nominal",
            "scale": shared_color_scale,
            "legend": legend_spec,
        }

    def color_no_legend():
        return {
            "field": "agent", "type": "nominal",
            "scale": shared_color_scale,
            "legend": None,
        }

    x_enc = {
        "field": "k", "type": "quantitative",
        "title": "Steps observed",
        "scale": {"domain": [0, 52]},
        "axis": {"values": [0, 10, 20, 30, 40, 50]},
    }
    y_mean_enc = {
        "field": "auc_mean", "type": "quantitative",
        "title": "AUC",
        "scale": {"domain": [0.3, 0.85]},
        "axis": {"values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    }

    spec = {
        "config": _THEME_CONFIG,
        "title": {
            "text": "Outcome predictability from first k steps",
            "anchor": "start",
            "offset": 10,
        },
        "width": CHART_WIDTH,
        "height": CHART_HEIGHT,
        "padding": {
            "top": 20,
            "bottom": 20,
            "left": 20,
            "right": right_padding,
        },
        "resolve": {"scale": {"color": "shared"}},
        "layer": [
            # Layer 0: CI bands — CARRIES LEGEND (workaround: legend on area/y2 layer)
            {
                "data": {"values": df_agents_records},
                "mark": {"type": "area", "opacity": 0.12},
                "encoding": {
                    "x": x_enc,
                    "y": {
                        "field": "auc_lo", "type": "quantitative",
                        "scale": {"domain": [0.3, 0.85]},
                        "title": "AUC",
                        "axis": {"values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
                    },
                    "y2": {"field": "auc_hi"},
                    "color": color_with_legend(),
                },
            },
            # Layer 1: agent lines (no legend — line+y2 combination suppresses legend)
            {
                "data": {"values": df_agents_records},
                "mark": {"type": "line", "strokeWidth": 2},
                "encoding": {
                    "x": x_enc,
                    "y": y_mean_enc,
                    "color": color_no_legend(),
                },
            },
            # Layer 2: agent points
            {
                "data": {"values": df_agents_records},
                "mark": {"type": "point", "filled": True, "size": 40},
                "encoding": {
                    "x": x_enc,
                    "y": y_mean_enc,
                    "color": color_no_legend(),
                },
            },
            # Layer 3: chance dashed line
            {
                "data": {"values": df_chance_records},
                "mark": {"type": "line", "strokeWidth": 1.5, "strokeDash": [4, 4]},
                "encoding": {
                    "x": x_enc,
                    "y": y_mean_enc,
                    "color": color_no_legend(),
                },
            },
        ],
    }

    return spec


def save_spec(spec: dict, scale_factor: int = 2) -> None:
    """Render spec to PNG via vl_convert."""
    png_bytes = vlc.vegalite_to_png(vl_spec=spec, scale=scale_factor)
    with open(OUT_PATH, "wb") as f:
        f.write(png_bytes)
    print(f"Saved chart to {OUT_PATH} ({len(png_bytes):,} bytes)")


# ---------------------------------------------------------------------------
# Vision-LLM verification via OpenRouter
# ---------------------------------------------------------------------------
def verify_legend_visible(image_path: str) -> tuple[bool, str]:
    """Return (legend_visible, explanation) using vision-LLM."""
    import urllib.request

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at this chart carefully. "
                            "Is the legend fully visible? (It may appear on the right side or inside.) "
                            "I expect to see ALL of these agent labels: "
                            "Claude-3, Claude-3.5, GPT-4, GPT-4o, Claude-3.7-thinking, Claude-4, "
                            "DARS+R1, Agentless+Claude-3.5, Moatless+V3, and 'Chance (AUC = 0.5)'. "
                            "Is any label clipped or cut off at the image edge? "
                            "Respond FIRST with exactly one word: VISIBLE (all labels fully readable) "
                            "or CLIPPED (any label cut off or missing). "
                            "Then list which labels you can see."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 400,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://thetaste.ai",
            "X-Title": "bidirect-align-figures",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        first_word = text.split()[0].upper().rstrip(".,:")
        visible = (first_word == "VISIBLE")
        return visible, text
    except Exception as e:
        print(f"Vision-LLM call failed: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# Main: save and verify, retry with adjusted parameters if needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Vary legend_x (distance from plot left edge) and right_padding
    attempts = [
        (520, 250),   # legend 20px right of plot edge, 250px right padding
        (520, 270),
        (530, 280),
        (510, 250),
        (540, 300),
    ]

    for attempt_num, (legend_x, right_pad) in enumerate(attempts, start=1):
        print(f"\n--- Attempt {attempt_num}: legend_x={legend_x}, right_padding={right_pad} ---")
        spec = build_spec(legend_x=legend_x, right_padding=right_pad)

        # Verify legend is present in SVG before even saving PNG
        svg = vlc.vegalite_to_svg(vl_spec=spec)
        legend_in_svg = "role-legend" in svg
        print(f"Legend in SVG: {legend_in_svg}")

        if not legend_in_svg:
            print("Legend not in SVG — skipping PNG save for this attempt")
            continue

        save_spec(spec, scale_factor=2)

        visible, explanation = verify_legend_visible(OUT_PATH)
        print(f"Vision-LLM verdict: {'VISIBLE' if visible else 'CLIPPED'}")
        print(f"Explanation:\n{explanation}")

        if visible:
            print(f"\nSuccess — legend is visible. Output at: {OUT_PATH}")
            break
    else:
        print(f"\nWarning: legend may still be clipped after all attempts.")
        print(f"Final output at: {OUT_PATH}")
