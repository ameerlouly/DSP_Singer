"""Check sizing policies without GUI dependencies, plus a live Qt test if installed."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'table_sizing.py').read_text(encoding='utf-8')
CONTROLLER = (ROOT / 'audio_review_controller.py').read_text(encoding='utf-8')
ASR = (ROOT / 'transcription_review_controller.py').read_text(encoding='utf-8')


def policy_methods():
    node = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.ClassDef))
    node.bases = []
    node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name not in ('__init__', 'eventFilter')]
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<table sizing>', 'exec'), env)
    return env['TableSizing']


class ResizableTablesTests(unittest.TestCase):
    def policy(self, equal=False, rows=1, width=1000):
        p = policy_methods()()
        p.equal, p.manual, p.applying, p.fitted = equal, False, False, False
        p.preferred = {}; p.timer = Mock(); p.header = Mock(); p.table = Mock()
        p.table.isVisible.return_value = True; p.table.columnCount.return_value = 5
        p.table.rowCount.return_value = rows; p.table.viewport.return_value.width.return_value = width
        p.header.sectionSize.return_value = 100
        return p

    def test_all_seven_tables_configured(self):
        names = {w.attrib['name'] for w in ET.parse(ROOT / 'main_window.ui').findall('.//widget[@class="QTableWidget"]')}
        self.assertEqual(len(names), 7)
        for name in names:
            self.assertIn(f'.table(self.ui.{name})', CONTROLLER + ASR)
        self.assertIn('table._table_sizing = TableSizing(table)', CONTROLLER)
        self.assertEqual(SOURCE.count('setSectionResizeMode(QHeaderView.ResizeMode.Interactive)'), 2)
        self.assertNotIn('ResizeMode.Stretch', SOURCE)
        self.assertIn('vertical.show()', SOURCE)

    def test_region_equal_until_manual_resize(self):
        p = self.policy(equal=True, width=1003); p.fit()
        self.assertEqual([c.args[1] for c in p.header.resizeSection.call_args_list], [201,201,201,200,200])
        p.header.resizeSection.reset_mock(); p.table.viewport.return_value.width.return_value = 750
        p.fit(); self.assertEqual([c.args[1] for c in p.header.resizeSection.call_args_list], [150]*5)
        p.section_resized(0, 150, 240)
        p.header.resizeSection.reset_mock(); p.fit(); p.schedule()
        p.header.resizeSection.assert_not_called(); p.timer.start.assert_not_called()

    def test_initial_fit_uses_preferred_width_and_fills_viewport(self):
        p = self.policy(); p.preferred = {1: 300}; p.fit()
        self.assertEqual([c.args[1] for c in p.header.resizeSection.call_args_list], [100,300,100,100,400])
        p.table.resizeColumnsToContents.assert_called_once()
        p.header.resizeSection.reset_mock(); p.fit(); p.schedule()
        p.header.resizeSection.assert_not_called(); p.timer.start.assert_not_called()

    def test_empty_and_hidden_tables_defer_fit(self):
        p = self.policy(rows=0); p.fit(); self.assertFalse(p.fitted)
        p.table.rowCount.return_value = 1; p.table.isVisible.return_value = False
        p.fit(); self.assertFalse(p.fitted)
        p.table.isVisible.return_value = True; p.fit(); self.assertTrue(p.fitted)

    def test_internal_resize_does_not_mark_manual(self):
        p = self.policy(); p.applying = True; p.section_resized(0, 100, 120)
        self.assertFalse(p.manual)

    def test_live_resize_if_available(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        try:
            from PyQt6.QtWidgets import QApplication, QHeaderView, QTableWidget, QTableWidgetItem
        except ImportError:
            self.skipTest('PyQt6 is not installed')
        from table_sizing import TableSizing
        app = QApplication.instance() or QApplication([])
        table = QTableWidget(2, 5); table.resize(900, 300)
        table._table_sizing = sizing = TableSizing(table); sizing.equal = True
        table.show(); app.processEvents(); sizing.fit()
        h = table.horizontalHeader(); v = table.verticalHeader()
        for i in range(5):
            self.assertEqual(h.sectionResizeMode(i), QHeaderView.ResizeMode.Interactive)
        self.assertEqual(v.sectionResizeMode(0), QHeaderView.ResizeMode.Interactive)
        self.assertTrue(v.isVisible())
        h.resizeSection(0, 230); v.resizeSection(0, 65)
        table.setItem(0,0,QTableWidgetItem('Updated')); table.resize(1100,300)
        sizing.schedule(); app.processEvents(); sizing.fit()
        self.assertEqual(h.sectionSize(0),230); self.assertEqual(v.sectionSize(0),65)
        table.close()


if __name__ == '__main__':
    unittest.main()
