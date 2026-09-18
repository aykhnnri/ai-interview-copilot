"""Small shared widgets."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import messages as msg
from . import theme

#: Provider state -> (dot colour, Azerbaijani caption).  Colours come from the
#: theme so the palette is defined in exactly one place.
STATE_STYLES: dict[str, tuple[str, str]] = {
    "connected": (theme.STATE_COLOURS["connected"], msg.STATUS_CONNECTED),
    "ready": (theme.STATE_COLOURS["ready"], msg.STATUS_CONNECTED),
    "generating": (theme.STATE_COLOURS["generating"], msg.STATUS_GENERATING),
    "connecting": (theme.STATE_COLOURS["connecting"], msg.STATUS_CONNECTING),
    "disconnected": (theme.STATE_COLOURS["disconnected"], msg.STATUS_DISCONNECTED),
    "error": (theme.STATE_COLOURS["error"], msg.STATUS_ERROR),
}


class StatusIndicator(QWidget):
    """A dot plus the provider name.

    In `compact` form the state word is dropped from the row and carried by the
    dot colour and the tooltip instead.  Two providers then cost one short line
    of chrome rather than four, which is most of what made the old header busy.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self._compact = compact

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(10)
        self._title = QLabel(title)
        self._state = QLabel(msg.STATUS_DISCONNECTED)

        size = 11 if compact else 12
        self._title.setStyleSheet(
            f"color:{theme.TEXT_DIM}; font-size:{size}px; font-weight:500;"
        )
        self._state.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:{size}px;")

        layout.addWidget(self._dot)
        layout.addWidget(self._title)
        layout.addWidget(self._state)
        if not compact:
            layout.addStretch(1)
        # The label keeps its text (it is the readable state, and tests read it);
        # it is simply not shown when the row has to stay short.
        self._state.setVisible(not compact)
        self.set_state("disconnected")

    def set_title(self, title: str) -> None:
        self._title.setText(title)
        self._refresh_tooltip()

    def set_detail(self, detail: str) -> None:
        """Extra text for the tooltip only - e.g. the exact model id.

        The row shows the provider; the precise model is something you look up
        occasionally, so it does not earn permanent space in the window.
        """
        self._detail = detail
        self._refresh_tooltip()

    def set_state(self, state: str, tooltip: str = "") -> None:
        colour, caption = STATE_STYLES.get(state, (theme.TEXT_FAINT, state))
        self._dot.setStyleSheet(f"color:{colour}; font-size:13px;")
        self._state.setText(caption)
        self._title.setStyleSheet(
            f"color:{theme.TEXT if state in ('connected', 'ready') else theme.TEXT_DIM};"
            f" font-size:{11 if self._compact else 12}px; font-weight:500;"
        )
        self._explicit_tooltip = tooltip
        self._refresh_tooltip()
        self.setProperty("providerState", state)

    def _refresh_tooltip(self) -> None:
        explicit = getattr(self, "_explicit_tooltip", "")
        if explicit:
            self.setToolTip(explicit)
            return
        detail = getattr(self, "_detail", "")
        head = f"{self._title.text()} · {detail}" if detail else self._title.text()
        self.setToolTip(f"{head} — {self._state.text()}")


class Field(QWidget):
    """A label above a widget, used throughout the settings dialog."""

    def __init__(self, label: str, widget: QWidget, hint: str = "") -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        caption = QLabel(label)
        caption.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        layout.addWidget(caption)
        layout.addWidget(widget)
        if hint:
            note = QLabel(hint)
            note.setWordWrap(True)
            note.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:10px;")
            note.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(note)
        self.widget = widget
