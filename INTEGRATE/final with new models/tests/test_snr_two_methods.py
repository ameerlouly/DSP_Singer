"""Use known signal and noise levels to check the energy and WADA estimates.

Decision tests cover disagreements, measurement limits and the reviewer's ability to
keep a manual correction after the threshold changes.
"""

import copy,csv,json,tempfile,unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import snr_estimators as estimators
import clip_quality_assessment as snr

class TwoMethodSNR(unittest.TestCase):
    # Generate clean samples with a repeatable random seed so added noise is measurable.
    def speech(self,seed=4):
        rng=np.random.default_rng(seed);n=160000
        x=rng.gamma(.4,1,n)*rng.choice([-1.,1.],n)
        return x*.025/np.sqrt(np.mean(x*x)),rng
    # Use a mixture with known noise power to check the WADA estimate.
    def test_wada_known_gamma_speech_and_white_noise(self):
        clean,rng=self.speech()
        for target in (-5,0,10,20):
            x=clean+rng.normal(0,.025*10**(-target/20),len(clean))
            r=estimators.wada_details(x)
            self.assertEqual(r['status'],'ok')
            self.assertLess(abs(r['value_db']-target),2.5,(target,r))
    # Vary gain and DC offset and check that WADA accounts for the whole waveform.
    def test_wada_gain_dc_invariance_and_all_samples_accounted(self):
        clean,rng=self.speech(8);x=(clean+rng.normal(0,.008,len(clean)))[:104000]
        a=estimators.wada_details(x);b=estimators.wada_details(x*1e-5+.1)
        self.assertAlmostEqual(a['value_db'],b['value_db'],places=2)
        self.assertEqual(sum(p['samples'] for p in a['blocks']),len(x))
        self.assertEqual(len(a['blocks']),2)
    # Compare the pause-energy result with the known SNR of the mixture.
    def test_energy_uses_pause_frames_and_matches_known_snr(self):
        rng=np.random.default_rng(9);rate=16000;n=rate*10
        clean=np.zeros(n);clean[rate//2:-rate//2]=rng.normal(0,.03,n-rate)
        for sigma in (.003,.01,.05):
            noise=rng.normal(0,sigma,n);observed=clean+noise
            expected=10*np.log10(np.mean(clean[rate//2:-rate//2]**2)/np.mean(noise[rate//2:-rate//2]**2))
            r=estimators.energy_details(observed,pause_intervals=[[0,.5],[9.5,10.]])
            self.assertEqual(r['status'],'ok');self.assertLess(abs(r['value_db']-expected),1.)
            self.assertGreaterEqual(r['noise_frames'],10)
    # Change the noise floor and check that the result reports the variation.
    def test_noise_profile_can_track_observed_changes_but_flags_variability(self):
        rng=np.random.default_rng(10);n=160000
        clean=np.zeros(n);clean[8000:-8000]=rng.normal(0,.04,n-16000)
        noise=rng.normal(0,1,n)*np.sqrt(np.linspace(.001**2,.01**2,n))
        result=estimators.energy_details(clean+noise,pause_intervals=[[0,.5],[9.5,10.]])
        self.assertEqual(result['status'],'ok');self.assertGreater(result['noise_variation_db'],6)
        self.assertTrue(any('varies' in note for note in result['notes']))
        self.assertGreater(len(result['noise_anchors']),1)
    # Without a usable noise measurement there is no basis for a high SNR reading.
    def test_no_measured_noise_floor_does_not_invent_high_snr(self):
        x=np.zeros(160000);x[8000:-8000]=.1*np.sin(np.arange(144000)*.1)
        r=estimators.energy_details(x,pause_intervals=[[0,.5],[9.5,10.]])
        self.assertIsNone(r['value_db']);self.assertEqual(r['noise_frames'],0)
        self.assertIsNone(estimators.energy_details(x,pause_intervals=[])['value_db'])
    # Give the estimators invalid samples and badly formed pause intervals.
    def test_invalid_audio_and_pause_metadata(self):
        for x in (np.zeros(16000),np.ones(16000),np.full(16000,np.nan),np.array([.1]),np.zeros((16000,2))):
            self.assertIsNone(estimators.energy_details(x)['value_db'])
            self.assertIsNone(estimators.wada_details(x)['value_db'])
        clean,_=self.speech()
        self.assertIsNone(estimators.energy_details(clean,pause_intervals=[[0,1000]])['value_db'])
    # A value at the end of the WADA table only bounds the true estimate.
    def test_censored_wada_is_not_used_as_an_exact_decision_value(self):
        x=np.sin(np.arange(160000)*.07)*.1
        w=estimators.wada_details(x)
        self.assertEqual(w['status'],'range_limited')
        item=dict(analysis_version=2,content_label='speech',energy_snr_db=22.,wada_snr_db=w['value_db'],wada_details=w,
            energy_details=dict(status='ok',noise_source='reviewed_part4_pause_context'))
        snr.decision(item,20,'conservative','Listening check')
        self.assertEqual(item['snr_db'],22.);self.assertFalse(item['review_required']);self.assertTrue(item['quality_warning'])
        snr.decision(item,20,'wada','Listening check');self.assertIsNone(item['snr_db'])
    # Build a clip with fixed quality scores so the decision rules can be checked
    # independently.
    def record(self):
        return snr.decision(dict(id='clip',file='clip.wav',analysis_version=2,content_label='speech',
            energy_snr_db=25.,wada_snr_db=15.,energy_details=dict(status='ok',noise_source='reviewed_part4_pause_context'),
            wada_details=dict(status='ok',bound=None),audio_sha256='controlled',override_by='',override_reason=''),20,'conservative','Validated on a listening sample')
    # Make the estimators disagree and inspect the policy and review flag.
    def test_decision_rule_and_disagreement_are_explicit(self):
        r=self.record();self.assertEqual((r['snr_db'],r['automatic_quality']),(15.,'noisy'))
        self.assertFalse(r['review_required']);self.assertTrue(r['quality_warning'])
        snr.decision(r,20,'energy','Listening check');self.assertEqual(r['automatic_quality'],'clean')
        self.assertFalse(r['review_required']);self.assertTrue(r['quality_warning'])
    # A valid listening correction should survive a new threshold or recalculation.
    def test_threshold_changes_and_recalculation_preserve_human_correction(self):
        r=self.record();snr.set_quality(r,'clean','Reviewer','Background acceptable after listening')
        changed=snr.reclassify([r],30,'conservative','Different calibrated threshold')[0]
        self.assertEqual(changed['quality'],'clean');self.assertEqual(changed['automatic_quality'],'noisy')
        self.assertEqual(changed['review_status'],'Manually corrected')
        fresh=self.record();snr.preserve_override(fresh,r);self.assertEqual(fresh['quality'],'clean')
        other=self.record();other['audio_sha256']='changed';snr.preserve_override(other,r);self.assertEqual(other['quality'],'noisy')
        snr.set_quality(changed,None,'Reviewer','Restore automatic');self.assertEqual(changed['quality'],'noisy')
        self.assertFalse(changed['override_by'])
    # Read the saved report back to check the measurements and the reviewer's reason.
    def test_reports_keep_both_values_policy_justification_and_override(self):
        r=self.record();snr.set_quality(r,'clean','Reviewer','Listened')
        with tempfile.TemporaryDirectory() as d:
            _,no_reason,_=snr.export_report([r],20,d)
            self.assertFalse(json.loads(Path(no_reason).read_text())['threshold_justification_provided'])
            html,json_file,csv_file=snr.export_report([r],20,d,justification='Listening review <script>unsafe</script>')
            payload=json.loads(Path(json_file).read_text())
            self.assertEqual(payload['methods'],['pause_energy','wada_snr']);self.assertEqual(payload['second_method'],'wada_snr')
            self.assertEqual(payload['segments'][0]['quality'],'clean')
            with Path(csv_file).open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual((rows[0]['energy_snr_db'],rows[0]['wada_snr_db']),('25.0','15.0'))
            self.assertIn('&lt;script&gt;',Path(html).read_text());self.assertNotIn('<script>unsafe',Path(html).read_text())
    # Older results with one estimator should remain identifiable when loaded.
    def test_legacy_data_remains_marked_single_method(self):
        r=dict(id='old',file='old.wav',snr_db=18.,quality='noisy',automatic_quality='noisy',content_label='speech')
        new=snr.reclassify([r],20,'energy','Legacy check')[0]
        self.assertFalse(new['review_required']);self.assertTrue(new['quality_warning']);self.assertIsNone(new['wada_snr_db'])
        self.assertTrue(any('Legacy' in text for text in new['review_reasons']))

if __name__=='__main__':unittest.main()
