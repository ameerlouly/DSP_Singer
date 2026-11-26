"""Approve All includes duration-valid automatic boundary-review candidates."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import segment_review as review


def candidate(i, start, duration, *, flagged=False, status='pending'):
    return dict(id=f'auto-{i:05d}', original=dict(start=start, end=start+duration, duration=duration,
        manual_boundary_review_required=flagged), label='speech', status=status,
        source_start=start, source_end=start+duration, source_mode='original_candidate',
        candidate_file=f'{i}.wav', reviewed_by=None)


def session():
    data = dict(source=dict(duration_seconds=150), changes=[], segments=[
        candidate(1, 0, 5), candidate(2, 6, 7.64, flagged=True),
        candidate(3, 15, 15, flagged=True), candidate(4, 31, 4.99),
        candidate(5, 37, 15.01, flagged=True), candidate(6, 54, 8, status='excluded'),
        candidate(7, 63, 8, flagged=True), candidate(8, 74, 8, status='approved'),
        dict(id='manual-01', original=None, label='mixed', status='pending',
             source_start=83,source_end=91, source_mode='continuous_source', reviewed_by=None)],
        unresolved=[dict(id='unresolved-00001',status='pending', original=dict(start=100,end=110)),
                    dict(id='gap-00001',status='pending', original=dict(start=110,end=112))])
    data['segments'][6]['original']['source_spans'] = [dict(start=63,end=67,clip_start=0,clip_end=4),
                                                       dict(start=68,end=72,clip_start=4,clip_end=8)]
    return data


class BatchApprovalTests(unittest.TestCase):
    def test_select_and_approve_valid_candidates(self):
        data=session(); before=copy.deepcopy(data)
        rows, skipped=review.batch_approval_plan(data)
        self.assertEqual(rows,[0,1,2,8]);self.assertEqual([i for i,_ in skipped],[3,4,6])
        self.assertEqual(data,before)
        self.assertEqual(review.approve_segments(data, rows, 'Reviewer', approval_method='batch_approve_all'),4)
        for i in rows:
            item=data['segments'][i]
            self.assertEqual(item['status'],'approved')
            self.assertEqual(item['pdf_boundary_review']['approval_method'],'batch_approve_all')
            self.assertEqual(item['reviewed_by'],'Reviewer')
            review.require_pdf_review(data,item)
        for i in [3,4,5,6,7]:self.assertEqual(data['segments'][i],before['segments'][i])
        self.assertEqual(data['unresolved'],before['unresolved'])
        self.assertEqual(len(data['changes']),4)
        self.assertEqual(review.batch_approval_plan(data)[0],[])

    def test_button_calls_batch_once_and_respects_cancel(self):
        # Execute the actual controller method with fake widgets; no Qt install needed.
        tree=ast.parse((Path(__file__).resolve().parents[1]/'audio_review_controller.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='approve_all')
        qt=SimpleNamespace(StandardButton=SimpleNamespace(Yes=1,No=2),question=Mock(return_value=2))
        env=dict(review=review,QMessageBox=qt)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<controller method>','exec'),env)
        data=session();before=copy.deepcopy(data)
        workspace=SimpleNamespace(session=data,window=None,reviewer=lambda:'Reviewer',modified_session=Mock(),status=Mock())
        env['approve_all'](workspace)
        self.assertEqual(data,before);workspace.modified_session.assert_not_called()
        qt.question.return_value=1
        env['approve_all'](workspace)
        self.assertEqual(qt.question.call_count,2)
        workspace.modified_session.assert_called_once_with([0,1,2,8])
        self.assertEqual([data['segments'][i]['status'] for i in [0,1,2,8]], ['approved']*4)
        self.assertIn('3 invalid', workspace.status.call_args.args[0])

if __name__=='__main__':unittest.main()
