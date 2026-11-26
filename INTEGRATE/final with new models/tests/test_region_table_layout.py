"""The region table is five initially equal-width, adjustable columns without changing review data."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'audio_review_controller.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)


def method(name, env):
    node = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<region table>', 'exec'), env)
    return env[name]


class RegionTableTests(unittest.TestCase):
    def test_five_headers(self):
        table = ET.parse(ROOT / 'main_window.ui').find('.//widget[@name="regionTable"]')
        self.assertEqual([c.find('property/string').text for c in table.findall('column')],
                         ['Start (s)', 'End (s)', 'Label', 'Automatic label', 'Auto confidence'])

    def test_equal_width_only_region_table(self):
        table = Mock()
        method('equal_columns', {})(table)
        self.assertTrue(table._table_sizing.equal)
        table._table_sizing.schedule.assert_called_once_with()
        self.assertEqual(SOURCE.count('self.equal_columns('), 1)
        self.assertIn('self.equal_columns(self.ui.regionTable)', SOURCE)

    def test_rows_keep_confidence_and_underlying_review_flags(self):
        records = [dict(start=0, end=9, label='music', auto_label='music', auto_confidence=.675, needs_review=True),
                   dict(start=9, end=10, label='mixed', auto_label='mixed', auto_confidence=.718, manual_override=True)]
        before = copy.deepcopy(records)
        table = Mock(); table.columnCount.return_value = 5
        workspace = SimpleNamespace(ui=SimpleNamespace(regionTable=table), regions=records, update_cells=Mock())
        method('region_rows', {'confidence_text': lambda r: f"{r['auto_confidence']:.3f} ({r['auto_confidence']:.1%})"})(workspace, 0, 2)
        rows = [call.args[2] for call in workspace.update_cells.call_args_list]
        self.assertEqual(rows, [['0.0000', '9.0000', 'music', 'music', '0.675 (67.5%)'],
                                ['9.0000', '10.0000', 'mixed', 'mixed', '0.718 (71.8%)']])
        self.assertEqual(records, before)
        self.assertEqual(table.item.call_count, 10)

    def test_live_qt_widths_if_available(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        try:
            from PyQt6 import uic
            from PyQt6.QtWidgets import QApplication, QHeaderView
        except ImportError:
            self.skipTest('PyQt6 is not installed')
        app = QApplication.instance() or QApplication([])
        window = uic.loadUi(str(ROOT / 'main_window.ui'))
        from table_sizing import TableSizing
        table = window.regionTable
        table._table_sizing = TableSizing(table)
        method('equal_columns', {})(table)
        self.assertEqual(table.columnCount(), 5)
        for col in range(5):
            self.assertEqual(table.horizontalHeader().sectionResizeMode(col), QHeaderView.ResizeMode.Interactive)
        window.close()


if __name__ == '__main__':
    unittest.main()
