import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import segment_review as review
import corpus_review_and_export as corpus
from activity_history import events_only
from test_batch_approval import candidate


class ActivityHistoryTests(unittest.TestCase):
    def test_approvals_and_label_only_are_not_logged_but_real_cut_is(self):
        data = dict(source=dict(duration_seconds=60), changes=[], segments=[candidate(1,0,8)],
                    unresolved=[dict(id='unresolved-1', label='unclassified', status='pending', original=dict(start=10,end=30))])
        review.approve_segment(data,0,'Human')
        review.require_pdf_review(data,data['segments'][0])
        self.assertEqual(data['changes'],[])
        review.label_unresolved(data,0,'speech','Human')
        review.mark_unresolved_reviewed(data,0,'Human')
        self.assertEqual(data['unresolved'][0]['status'],'reviewed')
        self.assertEqual(data['changes'],[])
        review.label_unresolved(data,0,'speech','Human',start=12,end=24)
        self.assertEqual([e['action'] for e in data['changes']],['label_or_trim_unresolved'])
        self.assertEqual(len(data['changes'][0]['after']['preserved_leftovers']),2)
        i=review.add_from_unresolved(data,0,12,20,'speech','Human')
        review.approve_segment(data,i,'Human')
        self.assertEqual([e['action'] for e in data['changes']],['label_or_trim_unresolved','add_clip_from_unresolved'])

    def test_transcript_log_stores_model_and_text_only(self):
        log=[]
        row=dict(segment_id='s1',status='needs_review',final_transcript='',source_metadata={'large':'x'*100000})
        before=copy.deepcopy(row)
        row.update(status='accepted',final_transcript='مرحبا',human_override=True)
        corpus.audit(log,'choose_asr_model','Editor',before,row,model='b')
        self.assertEqual(log[0]['model'],'Cohere')
        self.assertNotIn('source_metadata',log[0]['before'])
        self.assertNotIn('human_changes',log[0]['after'])
        self.assertEqual(log[0]['after']['final_transcript'],'مرحبا')
        before=copy.deepcopy(row)
        corpus.audit(log,'transcript_review','Editor',before,row)
        self.assertEqual(len(log),1)
        row['final_transcript']='أهلا'
        corpus.audit(log,'transcript_review','Editor',before,row)
        self.assertEqual(len(log),2)

    def test_legacy_export_excludes_approvals_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);wav=root/'source.wav';sf.write(wav,np.zeros(16*16000,dtype=np.int16),16000,subtype='PCM_16')
            approval=dict(action='approve_segment',item_id='s1',reviewer='Old',time_utc='2026-11-01',before={'segment_id':'s1'},after={'segment_id':'s1'})
            reviewed=dict(action='mark_unresolved_reviewed',item_id='u1',reviewer='Old',time_utc='2026-11-01')
            choice=dict(action='transcript_review',reviewer='Editor',time_utc='2026-11-02',before={'segment_id':'s1','final_transcript':''},after={'segment_id':'s1','final_transcript':'مرحبا'})
            cut=dict(action='add_clip_from_unresolved',item_id='s1',reviewer='Editor',time_utc='2026-11-01',after={'id':'s1'})
            records=[dict(segment_id=ident,start=a,end=b,label='speech',source_mode='candidate_preview',segmentation_status='pending',source_origin=str(wav),status='accepted',transcript_a='مرحبا',transcript_b='مرحبا',final_transcript='مرحبا',audio_pieces=[dict(path=str(wav),start=a,end=b)],human_changes=history,quality_history=[dict(reviewer='Old',before='noisy',after='clean')]) for ident,a,b,history in [('s1',0,8,[approval,cut]),('s2',8,16,[approval,reviewed])]]
            manifest={'audio':str(wav),'source_review':{'step4':{'segments':[],'unresolved':[],'changes':[approval,reviewed,cut]},'region_changes':[{'action':'auto_approve_region'}]}}
            original=copy.deepcopy(manifest)
            for automatic in (False,True):
                result=corpus.export_corpus(records,manifest,[approval,choice],root/'out',True,automatic=automatic)
                out=Path(result['output_dir'])
                with (out/'review_compatible.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                self.assertEqual([e['action'] for e in json.loads(rows[0]['change_log'])],['add_clip_from_unresolved','transcript_review'])
                self.assertEqual(rows[1]['change_log'],'[]');self.assertEqual(rows[1]['changed_by'],'');self.assertEqual(rows[1]['changed_at'],'')
                self.assertEqual([r['final_transcript'] for r in rows],['مرحبا','مرحبا'])
                self.assertEqual([r['status'] for r in rows],['accepted','accepted'])
                audit=json.loads((out/'corpus_audit.json').read_text());self.assertEqual(audit['audit_log'],[choice]);self.assertTrue(audit['segments'][0]['quality_history'])
                source=json.loads((out/'source_review.json').read_text());self.assertEqual(source['step4']['changes'],[cut]);self.assertEqual(source['region_changes'],[])
                self.assertEqual(manifest,original)
                self.assertEqual(len(list(out.glob('*.wav'))),2)

if __name__=='__main__':unittest.main()
