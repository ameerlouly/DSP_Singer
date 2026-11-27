"""Reviewed unresolved labels must not be mistaken for approved clips."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import test_ignored_gaps
import corpus_review_and_export as corpus

ROOT = Path(__file__).resolve().parents[1]

class PartialDefaultTests(unittest.TestCase):
    def test_manual_partial_default_in_custom_ui(self):
        ui = ET.parse(ROOT / 'main_window.ui')
        box = ui.find(".//widget[@name='allowPartialExport']")
        self.assertEqual(box.find("property[@name='checked']/bool").text, 'true')

    def test_all_reviewed_but_unresolved_can_export_complete_mode(self):
        # Reuse fixture with a valid approved clip and an audit-only gap.
        fixture = test_ignored_gaps.GapExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.session['unresolved'].append(dict(id='unresolved-reviewed-1', category='unresolved',
            label='speech', status='reviewed', reviewed_by='Operator', original=dict(start=10,end=22)))
        before = copy.deepcopy(fixture.session)
        manifest = fixture.manifest()
        with patch.object(corpus, 'compare'):
            result = corpus.export_corpus([fixture.record], manifest, [], fixture.root/'export', False)
        self.assertTrue(result['partial'])
        self.assertEqual(result['clips'],1)
        out=Path(result['output_dir'])
        self.assertTrue((out/'review_compatible.csv').is_file())
        self.assertTrue((out/'corpus_report.html').is_file())
        meta=json.loads((out/'corpus.json').read_text())['metadata']
        self.assertEqual(meta['pending_boundary_segment_ids'],[])
        self.assertEqual(meta['source_unresolved_span_count'],1)
        self.assertEqual(meta['export_mode'],'complete_with_unresolved')
        self.assertEqual(meta['unresolved_export_segment_ids'],[])
        self.assertIn('Complete-mode export with unresolved intervals',(out/'corpus_report.html').read_text())
        self.assertEqual(fixture.session,before)

if __name__=='__main__':unittest.main()
