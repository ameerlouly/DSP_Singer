"""Initial table layouts with mouse-adjustable columns and rows.

Interactive headers never re-fit after a user resize. Region columns share the
viewport equally until the first manual width change. Other tables fit once,
using a bounded content sample, then retain their widths across data refreshes.
"""
from PyQt6.QtCore import QObject, QEvent, QTimer
from PyQt6.QtWidgets import QHeaderView


class TableSizing(QObject):
    def __init__(self, table):
        super().__init__(table)
        self.table = table
        self.equal = False
        self.fitted = False
        self.manual = False
        self.applying = False
        self.preferred = {}
        self.header = table.horizontalHeader()
        self.header.setStretchLastSection(False)
        self.header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.header.setResizeContentsPrecision(48)
        self.header.setMinimumSectionSize(48)
        vertical = table.verticalHeader()
        vertical.setStretchLastSection(False)
        vertical.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        vertical.show()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.fit)
        self.header.sectionResized.connect(self.section_resized)
        table.viewport().installEventFilter(self)
        self.schedule()

    def schedule(self):
        if not self.manual and (self.equal or not self.fitted):
            self.timer.start(0)

    def section_resized(self, index, old, new):
        if not self.applying and old != new:
            self.manual = True
            self.timer.stop()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.Resize):
            self.schedule()
        return super().eventFilter(watched, event)

    def fit(self):
        table = self.table
        if self.manual or (self.fitted and not self.equal) or not table.isVisible():
            return
        count = table.columnCount()
        available = table.viewport().width()
        if not count or available <= 0:
            return
        # Fit non-region tables only after data exists, not on an empty startup.
        if not self.equal and not table.rowCount():
            return
        self.applying = True
        try:
            if self.equal:
                # Keep the initial equal distribution, with horizontal scrolling
                # on very narrow windows rather than unreadably small headings.
                base, remainder = divmod(max(available, count * 90), count)
                widths = [base + (i < remainder) for i in range(count)]
            else:
                table.resizeColumnsToContents()  # one shot; mode stays Interactive
                widths = [self.preferred.get(i, min(360, max(85, self.header.sectionSize(i))))
                          for i in range(count)]
                widths[-1] += max(0, available - sum(widths))
            for i, width in enumerate(widths):
                self.header.resizeSection(i, width)
            self.fitted = True
        finally:
            self.applying = False
