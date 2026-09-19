"""A tiny SVG drawing kit for the workshop diagrams.

Why hand-rolled instead of Mermaid or Graphviz?

* **No build step for students.** The output is a plain ``.svg`` committed to
  the repo. It renders in JupyterLab, on GitHub, and on the docs site with no
  extension, no kernel, and no network.
* **One visual language.** Every box, arrow and label comes from here, so
  eighteen diagrams drawn over an afternoon still look like a set.

**Colour strategy - read this before changing it.** Every colour is written
twice:

1. as a literal ``fill``/``stroke`` **presentation attribute** on the element,
   using the light palette;
2. as a CSS rule inside a ``prefers-color-scheme: dark`` media query, keyed on
   the element's class.

CSS beats presentation attributes, so a browser in dark mode gets the dark
palette, while a renderer that ignores CSS entirely still produces a correct
light diagram. An earlier version used CSS custom properties (``var(--bg)``)
for this and rendered **solid black** in every tool that does not implement
variables - including nbconvert's PDF path. Do not reintroduce them.

Coordinates are plain SVG user units with the origin at the top left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

LIGHT = {
    "bg": "#ffffff", "fg": "#1f2328", "muted_fg": "#656d76",
    "stroke": "#d0d7de", "box": "#f6f8fa",
    "accent": "#0969da", "accent_bg": "#ddf4ff",
    "good": "#1a7f37", "good_bg": "#dafbe1",
    "warn": "#9a6700", "warn_bg": "#fff8c5",
    "bad": "#cf222e", "bad_bg": "#ffebe9",
}
DARK = {
    "bg": "#0d1117", "fg": "#e6edf3", "muted_fg": "#8b949e",
    "stroke": "#30363d", "box": "#161b22",
    "accent": "#58a6ff", "accent_bg": "#132e57",
    "good": "#3fb950", "good_bg": "#12321f",
    "warn": "#d29922", "warn_bg": "#3a2d0a",
    "bad": "#f85149", "bad_bg": "#3d1417",
}

# role -> (fill key, stroke key)
ROLE_COLOURS = {
    "default": ("box", "stroke"),
    "accent": ("accent_bg", "accent"),
    "good": ("good_bg", "good"),
    "warn": ("warn_bg", "warn"),
    "bad": ("bad_bg", "bad"),
    "muted": ("box", "stroke"),
    "ghost": (None, "stroke"),
}

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif')
MONO = ('ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace')

TEXT_CLASSES = {
    "title": (17, 700, "fg"),
    "sub": (12.5, 400, "muted_fg"),
    "lbl": (13.5, 600, "fg"),
    "small": (11.5, 400, "muted_fg"),
    "mono": (12, 400, "fg"),
}


def _dark_css() -> str:
    """The dark-mode override block. Only colours - never geometry."""
    rules = [
        f"svg .d-bg {{ fill: {DARK['bg']}; }}",
        f"svg .d-fg {{ fill: {DARK['fg']}; }}",
        f"svg .d-muted_fg {{ fill: {DARK['muted_fg']}; }}",
        f"svg .d-edge {{ stroke: {DARK['stroke']}; }}",
        f"svg .d-edge-a {{ stroke: {DARK['accent']}; }}",
        f"svg .d-head {{ fill: {DARK['stroke']}; }}",
        f"svg .d-head-a {{ fill: {DARK['accent']}; }}",
    ]
    for role, (fill_key, stroke_key) in ROLE_COLOURS.items():
        fill = "none" if fill_key is None else DARK[fill_key]
        rules.append(f"svg .d-{role} {{ fill: {fill}; stroke: {DARK[stroke_key]}; }}")
    for tone in ("accent", "good", "warn", "bad"):
        rules.append(f"svg .d-t-{tone} {{ fill: {DARK[tone]}; }}")
    body = "\n  ".join(rules)
    return f"@media (prefers-color-scheme: dark) {{\n  {body}\n}}"


def _markers() -> str:
    return (
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LIGHT["stroke"]}" class="d-head"/></marker>'
        f'<marker id="ah-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LIGHT["accent"]}" class="d-head-a"/></marker>'
    )


# The shared registry. Any module that imports `diagram` and decorates a
# function adds it to the set that `make_diagrams.py` renders.
DIAGRAMS: list = []


def diagram(func):
    """Register a diagram function. It must return ``(filename, svg, caption)``."""
    DIAGRAMS.append(func)
    return func


# Approximate advance width per character, as a fraction of font size. Good
# enough to wrap label text without shipping a font-metrics library.
_CHAR_W = 0.55



def wrap(text: str, width_px: float, font_px: float) -> list[str]:
    """Greedy word wrap sized for the sans-serif stack above."""
    max_chars = max(4, int(width_px / (font_px * _CHAR_W)))
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


@dataclass
class Canvas:
    width: int
    height: int
    parts: list[str] = field(default_factory=list)

    # -- primitives --------------------------------------------------------

    def raw(self, markup: str) -> Canvas:
        self.parts.append(markup)
        return self

    def text(self, x: float, y: float, content: str, cls: str = "lbl",
             anchor: str = "middle") -> Canvas:
        """Draw text. `cls` is a base style, optionally plus a tone.

        e.g. ``"small t-good"`` renders at small size in the "good" colour.
        """
        tokens = cls.split()
        base = tokens[0]
        tone = next((t[2:] for t in tokens[1:] if t.startswith("t-")), None)
        size, weight, colour_key = TEXT_CLASSES.get(base, TEXT_CLASSES["lbl"])
        family = MONO if base == "mono" else FONT
        colour = LIGHT[tone] if tone else LIGHT[colour_key]
        dark_cls = f"d-t-{tone}" if tone else f"d-{colour_key}"
        return self.raw(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{family}\' '
            f'font-size="{size}" font-weight="{weight}" fill="{colour}" '
            f'class="{dark_cls}" text-anchor="{anchor}">{escape(content)}</text>'
        )

    def box(self, x: float, y: float, w: float, h: float, label: str = "",
            sub: str = "", role: str = "default", radius: float = 9,
            label_cls: str = "lbl", mono: bool = False) -> Canvas:
        """A rounded rectangle with a wrapped, vertically centred label."""
        # Easy mistake: box(x, y, w, h, "label", "accent") puts the role in the
        # `sub` slot, so the diagram silently renders "accent" as a subtitle and
        # uses the default colour. Refuse it rather than ship it.
        if sub in ROLE_COLOURS and role == "default":
            raise ValueError(
                f"box(): sub={sub!r} is a role name - did you mean role={sub!r}? "
                "Pass it by keyword."
            )
        fill_key, stroke_key = ROLE_COLOURS.get(role, ROLE_COLOURS["default"])
        fill = "none" if fill_key is None else LIGHT[fill_key]
        dashes = ' stroke-dasharray="5 4"' if role in ("muted", "ghost") else ""
        self.raw(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{radius}" fill="{fill}" stroke="{LIGHT[stroke_key]}" '
            f'stroke-width="1.6"{dashes} class="d-{role}"/>'
        )
        if not label and not sub:
            return self

        cls = "mono" if mono else label_cls
        font = TEXT_CLASSES["mono" if mono else label_cls][0]
        lines = wrap(label, w - 16, font) if label else []
        sub_lines = wrap(sub, w - 16, 11.5) if sub else []

        line_h, sub_h = font + 3.5, 14
        total = len(lines) * line_h + (len(sub_lines) * sub_h if sub_lines else 0)
        cursor = y + h / 2 - total / 2 + font

        for line in lines:
            self.text(x + w / 2, cursor, line, cls)
            cursor += line_h
        if sub_lines:
            cursor += 1
            for line in sub_lines:
                self.text(x + w / 2, cursor, line, "small")
                cursor += sub_h
        return self

    def arrow(self, x1: float, y1: float, x2: float, y2: float,
              label: str = "", accent: bool = False, dashed: bool = False,
              label_dy: float = -7, label_cls: str = "small") -> Canvas:
        colour = LIGHT["accent"] if accent else LIGHT["stroke"]
        dark_cls = "d-edge-a" if accent else "d-edge"
        dashes = ' stroke-dasharray="5 4"' if dashed else ""
        marker = "ah-a" if accent else "ah"
        self.raw(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}" stroke-width="{2 if accent else 1.8}" fill="none"'
            f'{dashes} class="{dark_cls}" marker-end="url(#{marker})"/>'
        )
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, label_cls)
        return self

    def path(self, d: str, accent: bool = False, dashed: bool = False,
             arrow: bool = True) -> Canvas:
        colour = LIGHT["accent"] if accent else LIGHT["stroke"]
        dark_cls = "d-edge-a" if accent else "d-edge"
        dashes = ' stroke-dasharray="5 4"' if dashed else ""
        marker = (' marker-end="url(#%s)"' % ("ah-a" if accent else "ah")) if arrow else ""
        return self.raw(
            f'<path d="{d}" stroke="{colour}" stroke-width="{2 if accent else 1.8}" '
            f'fill="none"{dashes} class="{dark_cls}"{marker}/>'
        )

    def circle(self, cx: float, cy: float, r: float, role: str = "accent") -> Canvas:
        fill_key, stroke_key = ROLE_COLOURS.get(role, ROLE_COLOURS["default"])
        fill = "none" if fill_key is None else LIGHT[fill_key]
        return self.raw(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{LIGHT[stroke_key]}" stroke-width="1.6" class="d-{role}"/>'
        )

    def rule(self, x1: float, y1: float, x2: float, y2: float) -> Canvas:
        return self.raw(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{LIGHT["stroke"]}" stroke-width="1.8" class="d-edge"/>'
        )

    def title(self, content: str, sub: str = "", x: float = 24, y: float = 30) -> Canvas:
        self.text(x, y, content, "title", anchor="start")
        if sub:
            self.text(x, y + 19, sub, "sub", anchor="start")
        return self

    # -- output ------------------------------------------------------------

    def render(self, description: str = "") -> str:
        desc = f"<desc>{escape(description)}</desc>" if description else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img">'
            f"{desc}"
            f"<defs>{_markers()}</defs>"
            f"<style>{_dark_css()}</style>"
            f'<rect width="{self.width}" height="{self.height}" fill="{LIGHT["bg"]}" '
            f'class="d-bg"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )
