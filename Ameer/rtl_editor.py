"""Qt-based Arabic transcript editor with bidirectional text layout."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
    QVBoxLayout,
)

_qt_application: QApplication | None = None


class ArabicTextEditorDialog(QDialog):
    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(760, 420)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        self.editor = QTextEdit(self)
        self.editor.setFont(QFont("Segoe UI", 12))
        self.editor.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        text_option = QTextOption()
        text_option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        text_option.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.editor.document().setDefaultTextOption(text_option)
        self.editor.setPlainText(text)
        self.editor.moveCursor(QTextCursor.MoveOperation.End)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(buttons)
        self.editor.setFocus()


def edit_arabic_text(title: str, text: str) -> str | None:
    global _qt_application
    application = QApplication.instance()
    if application is None:
        _qt_application = QApplication([])
        application = _qt_application
    application.setQuitOnLastWindowClosed(False)

    dialog = ArabicTextEditorDialog(title, text)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.editor.toPlainText()
    return None
