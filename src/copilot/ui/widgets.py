"""Small shared widgets."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .. import messages as msg

#: Provider state -> (dot colour, Azerbaijani caption)
STATE_STYLES: dict[str, tuple[str, str]] = {
    "connected": ("#3fb950", msg.STATUS_CONNECTED),
    "ready": ("#3fb950", msg.STATUS_CONNECTED),
    "generating": ("#58a6ff", msg.STATUS_GENERATING),
    "connecting": ("#d29922", msg.STATUS_CONNECTING),
    "disconnected": ("#6e7681", msg.STATUS_DISCONNECTED),
    "error": ("#f85149", msg.STATUS_ERROR),
}


class StatusIndicator(QWidget):
    """A coloured dot, the provider/model name and a one-word state."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        self._title = QLabel(title)
        self._title.setStyleSheet("color:#e6e8ee; font-weight:600;")
        self._state = QLabel(msg.STATUS_DISCONNECTED)
        self._state.setStyleSheet("color:#8d94a5;")

        layout.addWidget(self._dot)
        layout.addWidget(self._title)
        layout.addWidget(self._state)
        layout.addStretch(1)
        self.set_state("disconnected")

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_state(self, state: str, tooltip: str = "") -> None:
        colour, caption = STATE_STYLES.get(state, ("#6e7681", state))
        self._dot.setStyleSheet(f"color:{colour}; font-size:14px;")
        self._state.setText(caption)
        self.setToolTip(tooltip or caption)
        self.setProperty("providerState", state)


class Field(QWidget):
    """A label above a widget, used throughout the settings dialog."""

    def __init__(self, label: str, widget: QWidget, hint: str = "") -> None:
        super().__init__()
        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        caption = QLabel(label)
        caption.setStyleSheet("color:#8d94a5; font-size:11px;")
        layout.addWidget(caption)
        layout.addWidget(widget)
        if hint:
            note = QLabel(hint)
            note.setWordWrap(True)
            note.setStyleSheet("color:#6e7681; font-size:10px;")
            note.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(note)
        self.widget = widget
