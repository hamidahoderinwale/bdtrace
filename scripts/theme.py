"""Shared Altair theme and color palette for all paper figures.

Project palette: five base colors plus two extended-cell agent colors.
Priority order for figure prominence: GREEN > BLUE > MAGENTA > COPPER > OLIVE.

    GREEN   #20A380  teal         — Claude-3.5, POSITIVE, structural
    BLUE    #5692E5  medium blue  — GPT-4, STRUCTURAL, benchmark primary
    MAGENTA #B4184F  crimson      — GPT-4o, HARD, warning/error
    COPPER  #CB4D20  terracotta   — Claude-3, warm accent, semantic baseline
    OLIVE   #585E53  dark olive   — NEUTRAL, annotation, support text
    INDIGO  #4658A0  slate-blue   — Claude-3.7-thinking (extended-thinking variant)
    VIOLET  #7A4FA8  deep purple  — Claude-4 (latest, top performer in the corpus)

INDIGO and VIOLET fill the hue gap between BLUE and MAGENTA so the two
extended-thinking-cell agents are distinguishable from the four base-cell
agents without claiming an existing slot.

Darker family variants for additional series:
    GREEN_D  #187860   BLUE_D    #3D7AD8   MAGENTA_D #8C1040
    COPPER_D #A03D18   INDIGO_D  #2E3D74   VIOLET_D  #5A347E

Import and call register() at the top of every figure script:

    from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE
    register()
"""
import altair as alt

# --- Project palette ---
GREEN   = "#20A380"  # teal       — Claude-3.5, positive, structural
BLUE    = "#5692E5"  # medium blue — GPT-4, structural
MAGENTA = "#B4184F"  # crimson    — GPT-4o, hard/warn
COPPER  = "#CB4D20"  # terracotta — Claude-3, warm accent
OLIVE   = "#585E53"  # dark olive — neutral, annotation, support
INDIGO  = "#4658A0"  # slate-blue — Claude-3.7-thinking (extended-thinking variant)
VIOLET  = "#7A4FA8"  # deep purple — Claude-4 (latest, top performer in this corpus)

# Darker family variants
GREEN_D   = "#187860"
BLUE_D    = "#3D7AD8"
MAGENTA_D = "#8C1040"
COPPER_D  = "#A03D18"
INDIGO_D  = "#2E3D74"
VIOLET_D  = "#5A347E"

# Legacy aliases (backward compatibility)
TEAL       = GREEN
RUST       = COPPER
ORANGE     = COPPER
VERMILLION = MAGENTA
PINK       = MAGENTA_D
SKY        = BLUE_D
YELLOW     = "#E8D84A"  # accent-only
GRAY       = OLIVE
NEAR_BLACK = "#1C1C2E"

# Semantic aliases for this paper
STRUCTURAL = BLUE     # FIM / edit-cert structural measures
SEMANTIC   = OLIVE    # semantic / embedding baselines (contrast)
HARD       = MAGENTA  # hard instances, failures, composition gap
POSITIVE   = GREEN    # correct, passing
NEUTRAL    = OLIVE    # neutral / background reference

# Benchmark aliases
LITE      = BLUE
VERIFIED  = GREEN
SWE_SMITH = COPPER

# Ordered list for categorical ranges (priority order)
CATEGORY = [GREEN, BLUE, MAGENTA, COPPER, OLIVE, GREEN_D, BLUE_D, MAGENTA_D]

# Stage taxonomy for action-sequence vignettes
STAGE_COLORS = {
    "Explore": BLUE,     # navigating / searching
    "Browse":  GREEN,    # reading files
    "Edit":    COPPER,   # writing / modifying
    "Test":    MAGENTA,  # running tests
    "Finish":  OLIVE,    # submit / other
}
STAGE_ORDER = ["Explore", "Browse", "Edit", "Test", "Finish"]

_FONT = "Inter, -apple-system, system-ui, sans-serif"

# Deprecated: do not add grid lines or reference lines to plots.
YREF = {"grid": False}
XREF = {"grid": False}

# Canonical agent identity — single source of truth for all figure scripts.
# Claude-3            = COPPER  (oldest Claude, lowest pass rate)
# Claude-3.5          = GREEN   (priority 1 — best Anthropic in base cell)
# GPT-4               = BLUE    (priority 2 — confident, targeted)
# GPT-4o              = MAGENTA (priority 3 — exploratory, higher token cost)
# Claude-3.7-thinking = INDIGO  (extended-thinking; slate-blue conveys "contemplative")
# Claude-4            = VIOLET  (latest, top performer in the corpus)
AGENT_COLORS = {
    "Claude-3":            COPPER,   # #CB4D20
    "Claude-3.5":          GREEN,    # #20A380
    "GPT-4":               BLUE,     # #5692E5
    "GPT-4o":              MAGENTA,  # #B4184F
    "Claude-3.7-thinking": INDIGO,   # #4658A0
    "Claude-4":            VIOLET,   # #7A4FA8
}
AGENT_ORDER = [
    "Claude-3", "Claude-3.5", "GPT-4", "GPT-4o",
    "Claude-3.7-thinking", "Claude-4",
]

# Canonical mapping from SWE-bench submission IDs to short agent names
AGENT_SHORT = {
    "20240402_sweagent_claude3opus":                "Claude-3",
    "20240402_sweagent_gpt4":                       "GPT-4",
    "20240620_sweagent_claude3.5sonnet":            "Claude-3.5",
    "20240728_sweagent_gpt4o":                      "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":   "Claude-4",
}


def register() -> None:
    """Register and enable the 'paper' Altair theme. Call once per script."""

    @alt.theme.register("paper", enable=True)
    def _paper_theme():
        return {
            "config": {
                "background": "#FFFFFF",
                "view": {
                    "stroke": None,
                    "continuousWidth": 500,
                    "continuousHeight": 320,
                },
                "axis": {
                    "labelFont": _FONT,
                    "titleFont": _FONT,
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "labelFontWeight": "normal",
                    "titleFontWeight": "normal",
                    "labelColor": "#444444",
                    "titleColor": "#222222",
                    "grid": False,
                    "domain": False,
                    "ticks": False,
                },
                "axisX": {"labelAngle": 0},
                "axisY": {"labelLimit": 380},
                "title": {
                    "font": _FONT,
                    "fontSize": 13,
                    "fontWeight": "normal",
                    "color": "#111111",
                    "anchor": "start",
                    "offset": 10,
                },
                "legend": {
                    "labelFont": _FONT,
                    "titleFont": _FONT,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "symbolSize": 80,
                    "padding": 8,
                },
                "range": {
                    "category": CATEGORY,
                },
                "mark": {"font": _FONT},
                "bar": {},
                "point": {"size": 60, "strokeWidth": 1.5},
                "line": {"strokeWidth": 2},
                "text": {"font": _FONT, "fontSize": 11, "color": "#333333"},
            }
        }
