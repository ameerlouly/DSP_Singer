"""V15: read-only final transcript review and safe next-segment navigation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'transcription_review_controller.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE)


def method(name):
    node=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'corpus':SimpleNamespace(is_preview=lambda r:r.get('segmentation_status')=='unresolved')}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<review method>','exec'),env)
    return env[name]


class ReviewNavigationTests(unittest.TestCase):
    def test_readonly_layout_and_actions(self):
        tree=ET.parse(ROOT/'main_window.ui')
        layout=tree.find('.//layout[@name="reviewPageLayout"]')
        direct=[item.find('widget').get('name') for item in layout.findall('item')]
        self.assertEqual(direct[2:4],['finalTable','finalTranscriptViewer'])
        self.assertEqual(len(layout.get('stretch').split(',')),len(direct))
        for name in ('finalEditorToggle','finalEditor','finalStart','finalEnd','finalLabel','finalQuality','finalStatus','finalReason','finalApply','finalSplit','finalMerge','finalAdd','finalDelete','finalRecalculate','finalSNRThreshold'):
            self.assertIsNone(tree.find(f'.//widget[@name="{name}"]'),name)
        for name in ('finalText','finalPlay','finalToASR','finalSaveReview','finalExport','allowPartialExport','processSummaryBox','asrNextSegment','asrSelectedStatus'):
            self.assertIsNotNone(tree.find(f'.//widget[@name="{name}"]'),name)
        self.assertEqual(tree.find('.//widget[@name="finalText"]/property[@name="readOnly"]/bool').text,'true')
        self.assertEqual(tree.find('.//widget[@name="asrFinalText"]/property[@name="readOnly"]/bool').text,'false')
        self.assertIn("'asrNextSegment':self.next_segment",SOURCE)
        self.assertNotIn('self.ui.finalText.textChanged.connect',SOURCE)

    def test_next_commits_draft_before_selection_and_scrolls(self):
        events=[]
        table=Mock();table.clearSelection.side_effect=lambda:events.append('clear')
        table.selectRow.side_effect=lambda i:events.append(('select',i))
        table.item.return_value='cell'
        w=SimpleNamespace(ui=SimpleNamespace(asrTable=table),records=[{}, {}, {}],p=SimpleNamespace(selected=lambda t:[1],status=Mock()),commit_drafts=lambda:events.append('commit'))
        method('next_segment')(w)
        self.assertEqual(events,['commit','clear',('select',2)])
        table.scrollToItem.assert_called_once_with('cell')

    def test_next_at_end_does_not_wrap_or_commit(self):
        table=Mock();w=SimpleNamespace(ui=SimpleNamespace(asrTable=table),records=[{}],p=SimpleNamespace(selected=lambda t:[0],status=Mock()),commit_drafts=Mock())
        method('next_segment')(w)
        w.p.status.assert_called_once_with('Last segment reached.')
        w.commit_drafts.assert_not_called();table.selectRow.assert_not_called()

    def test_next_failed_commit_preserves_selection(self):
        table=Mock();w=SimpleNamespace(ui=SimpleNamespace(asrTable=table),records=[{},{}],p=SimpleNamespace(selected=lambda t:[0]),commit_drafts=Mock(side_effect=ValueError('draft')))
        with self.assertRaisesRegex(ValueError,'draft'):method('next_segment')(w)
        table.clearSelection.assert_not_called();table.selectRow.assert_not_called()

    def test_next_without_selection_starts_first_row(self):
        table=Mock();w=SimpleNamespace(ui=SimpleNamespace(asrTable=table),records=[{}],p=SimpleNamespace(selected=lambda t:[]),commit_drafts=Mock())
        method('next_segment')(w);table.selectRow.assert_called_once_with(0)

    def test_final_review_displays_pending_and_approved_without_mutation(self):
        records=[dict(segment_id='a',start=1,end=9,status='accepted',segmentation_status='resolved',final_transcript='مرحبا'),dict(segment_id='b',start=9,end=30,status='needs_review',segmentation_status='unresolved',final_transcript='')]
        before=copy.deepcopy(records)
        selected=[0]
        ui=SimpleNamespace(finalTable=Mock(),finalText=Mock(),finalMapping=Mock(),finalPlay=Mock())
        w=SimpleNamespace(ui=ui,p=SimpleNamespace(selected=lambda t:selected),final_records=lambda:records,set_text=Mock())
        for i,expected in [(0,'Accepted'),(1,'Needs Review')]:
            selected[:]=[i];method('select_final')(w)
            w.set_text.assert_called_with(ui.finalText,records[i]['final_transcript'])
            self.assertIn(expected,ui.finalMapping.setText.call_args.args[0]);ui.finalPlay.setEnabled.assert_called_with(True)
        selected.clear();method('select_final')(w);ui.finalPlay.setEnabled.assert_called_with(False)
        self.assertEqual(records,before)

    def test_status_displays_selected_position_and_decision(self):
        r=dict(segment_id='b',status='difficult',segmentation_status='pending',word_difference=6)
        ui=SimpleNamespace(**{key:Mock() for key in ('asrSelectedStatus','asrComparison','asrCerA','asrCerB','asrAutomaticChoice','asrAutomaticReason')})
        w=SimpleNamespace(ui=ui,records=[{},r],row_by_id={'b':1})
        method('show_comparison')(w,r)
        line=ui.asrSelectedStatus.setText.call_args.args[0]
        self.assertIn('2/2',line);self.assertIn('Difficult',line);self.assertIn('Pending',line)

    def test_real_qt_layout_if_available(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        try:
            from PyQt6 import uic
            from PyQt6.QtWidgets import QApplication
        except ImportError:self.skipTest('PyQt6 is not installed')
        app=QApplication.instance() or QApplication([])
        window=uic.loadUi(str(ROOT/'main_window.ui'))
        self.assertTrue(window.finalText.isReadOnly());self.assertFalse(window.asrFinalText.isReadOnly())
        self.assertEqual(window.asrNextSegment.text(),'Next segment →')
        window.close()

if __name__=='__main__':unittest.main()
