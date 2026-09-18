"""The visual language: one neutral dark palette, defined once.

Strictly greyscale.  The only colour in the whole application is the error
tone, because a failed provider has to be distinguishable at a glance and grey
cannot carry that on its own.  Everything else - including what would normally
be a blue "primary" button - is a step on the same neutral ramp, so the window
reads as a quiet surface laid over the call rather than another app competing
for attention.
"""
from __future__ import annotations

# -- the ramp -------------------------------------------------------------
# Deliberately few steps.  More greys than this and the eye starts hunting for
# meaning in differences that carry none.
BG_WINDOW = "#0e0e10"      # behind everything
BG_CARD = "#161619"        # panels
BG_RAISED = "#1e1e22"      # inputs, the question block
BG_HOVER = "#2a2a30"       # button hover
BORDER = "#2b2b31"
BORDER_SOFT = "#202024"

TEXT = "#f2f2f5"           # the answer, headings
TEXT_DIM = "#9a9aa4"       # captions, secondary rows
TEXT_FAINT = "#65656f"     # hints, disabled, the in-flight partial

ACCENT = "#ececf0"         # near-white: the "primary" surface
ACCENT_HOVER = "#ffffff"
ACCENT_TEXT = "#121214"    # text printed on ACCENT
ERROR = "#c96f6f"          # the one colour in the app

#: Provider state -> dot colour.  Monochrome apart from the error tone.
STATE_COLOURS: dict[str, str] = {
    "connected": ACCENT,
    "ready": ACCENT,
    "generating": TEXT_DIM,
    "connecting": TEXT_FAINT,
    "disconnected": "#3a3a42",
    "error": ERROR,
}


def stylesheet(font_scale: float = 1.0) -> str:
    """The application stylesheet, with every font size scaled by `font_scale`.

    Scaling here rather than per widget keeps the type hierarchy intact at any
    size: the answer stays the largest thing on screen.
    """
    def px(size: float) -> str:
        return f"{max(8, round(size * font_scale))}px"

    return f"""
QMainWindow, QWidget#Root {{ background-color: {BG_WINDOW}; }}
QWidget {{ color: {TEXT}; }}

/* ---- structure ---- */
QFrame#Card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SOFT};
    border-radius: 12px;
}}
QFrame#Bar {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
}}
QFrame#Divider {{ background-color: {BORDER_SOFT}; border: none; }}

/* ---- type ---- */
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel#SectionTitle {{
    color: {TEXT_FAINT};
    font-size: {px(10)};
    font-weight: 600;
    letter-spacing: 1.4px;
}}
QLabel#Hint {{ color: {TEXT_FAINT}; font-size: {px(10)}; }}
QLabel#Meta {{ color: {TEXT_FAINT}; font-size: {px(10)}; }}

/* The question: present, but clearly subordinate to the answer. */
QLabel#Question {{
    color: {TEXT};
    font-size: {px(15)};
    font-weight: 600;
    line-height: 140%;
    padding: 12px 14px;
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
}}
QLabel#QuestionEmpty {{
    color: {TEXT_FAINT};
    font-size: {px(13)};
    font-weight: 400;
    font-style: italic;
    padding: 12px 14px;
    background-color: transparent;
    border: 1px dashed {BORDER};
    border-radius: 10px;
}}

/* ---- the answer: the reason the window exists ---- */
QTextEdit#Answer {{
    background-color: transparent;
    color: {TEXT};
    border: none;
    padding: 4px 2px;
    font-size: {px(16)};
    selection-background-color: {BG_HOVER};
    selection-color: {TEXT};
}}
QTextEdit#Transcript {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: none;
    padding: 2px;
    font-size: {px(11)};
    selection-background-color: {BG_HOVER};
}}
QTextEdit {{
    background-color: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px;
    selection-background-color: {BG_HOVER};
}}

/* ---- controls ---- */
QPushButton {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    font-size: {px(12)};
    font-weight: 500;
}}
QPushButton:hover:enabled {{ background-color: {BG_HOVER}; color: {TEXT}; }}
QPushButton:pressed:enabled {{ background-color: {BORDER}; }}
QPushButton:disabled {{ color: #4a4a52; border-color: {BORDER_SOFT}; }}
QPushButton:checked {{ background-color: {BG_HOVER}; color: {TEXT}; }}

QPushButton#Primary {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#Primary:hover:enabled {{ background-color: {ACCENT_HOVER}; }}
QPushButton#Primary:disabled {{
    background-color: {BG_RAISED}; color: #4a4a52; border-color: {BORDER_SOFT};
}}
QPushButton#Danger {{ color: {ERROR}; border-color: #40282a; }}
QPushButton#Danger:hover:enabled {{ background-color: #291b1d; color: {ERROR}; }}

/* Quiet icon-sized buttons for the top bar. */
QPushButton#Ghost {{
    border: none; background: transparent; color: {TEXT_FAINT};
    padding: 5px 9px; font-size: {px(12)};
}}
QPushButton#Ghost:hover:enabled {{ color: {TEXT}; background-color: {BG_HOVER}; }}
QPushButton#Ghost:checked {{ color: {TEXT}; background-color: {BG_HOVER}; }}

/* ---- inputs ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 9px;
    font-size: {px(12)};
    selection-background-color: {BG_HOVER};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {TEXT_FAINT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_RAISED}; color: {TEXT};
    border: 1px solid {BORDER}; selection-background-color: {BG_HOVER};
}}
QCheckBox {{ color: {TEXT_DIM}; font-size: {px(12)}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {BORDER}; background-color: {BG_RAISED};
}}
QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}

/* ---- audio level: a thin grey seam, not a progress bar ---- */
QProgressBar {{
    background-color: {BG_RAISED}; border: none;
    border-radius: 2px; height: 4px; text-align: center;
}}
QProgressBar::chunk {{ background-color: {TEXT_DIM}; border-radius: 2px; }}

/* ---- sliders ---- */
QSlider::groove:horizontal {{
    height: 4px; background: {BG_RAISED}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {TEXT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
}}

/* ---- chrome ---- */
QSplitter::handle {{ background-color: transparent; height: 6px; }}
QMenuBar {{ background-color: {BG_WINDOW}; color: {TEXT_DIM}; font-size: {px(12)}; }}
QMenuBar::item:selected {{ background-color: {BG_HOVER}; color: {TEXT}; }}
QMenu {{
    background-color: {BG_CARD}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background-color: {BG_HOVER}; }}
QTabWidget::pane {{ border: 1px solid {BORDER_SOFT}; border-radius: 10px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_FAINT};
    padding: 8px 15px; border: none; font-size: {px(12)};
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT_DIM}; }}
QGroupBox {{
    border: 1px solid {BORDER_SOFT}; border-radius: 10px;
    margin-top: 10px; padding-top: 10px; font-size: {px(12)};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 5px;
    color: {TEXT_FAINT}; font-size: {px(10)}; letter-spacing: 1.2px;
}}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; }}
QToolTip {{
    background-color: {BG_RAISED}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 5px 8px; border-radius: 6px;
}}
"""
