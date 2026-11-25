"""Change the label inside a larger region without losing the audio on either side.

relabel_range splits at the marked times and carries the original window evidence into
the resulting pieces.
"""

import copy
import math

RATE = 16000
LABELS = ('speech', 'music', 'mixed', 'silence')


def relabel_range(region, start, end, label):
    """Return replacement pieces and the index of the relabeled middle piece.

    Original inference windows retain their original bounds and model scores.
    A piece's displayed automatic scores use overlap-duration weighting; changing
    a human label does not invent a new model confidence.
    """
    if label not in LABELS:
        raise ValueError('Choose speech, music, mixed or silence.')
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError('Enter valid start/end times in seconds.') from exc
    if not math.isfinite(start + end):
        raise ValueError('Start/end times must be finite.')
    first, last = round(float(region['start']) * RATE), round(float(region['end']) * RATE)
    a, b = round(start * RATE), round(end * RATE)
    if not first <= a < b <= last:
        raise ValueError('The selected range must be inside the selected region, with start before end.')

    # Keep a piece of the original region with the analysis evidence that belongs to it.
    def piece(left, right, changed):
        item = copy.deepcopy(region)
        item.update(start=left / RATE, end=right / RATE, boundaries_modified=True)
        if 'analysis_windows' in region:
            windows = [copy.deepcopy(w) for w in region['analysis_windows']
                       if round(w['end'] * RATE) > left and round(w['start'] * RATE) < right]
            item['analysis_windows'] = windows
            item['window_count'] = len(windows)
            weighted = [(max(0, min(right, round(w['end'] * RATE)) - max(left, round(w['start'] * RATE))), w)
                        for w in windows]
            for key in ('confidence', 'auto_confidence', 'speech_score', 'music_score',
                        'vocal_score', 'voice_score', 'music_evidence_score', 'verified_voice_score'):
                valid = [(weight, float(w[key])) for weight, w in weighted if w.get(key) is not None]
                total = sum(weight for weight, score in valid)
                item[key] = sum(weight * score for weight, score in valid) / total if total else None
            names = {name for _, w in weighted for name in w.get('vocal_class_scores', {})}
            if 'vocal_class_scores' in item or names:
                item['vocal_class_scores'] = {}
                for name in sorted(names):
                    valid = [(weight, w['vocal_class_scores'][name]) for weight, w in weighted
                             if name in w.get('vocal_class_scores', {})]
                    item['vocal_class_scores'][name] = sum(weight * value for weight, value in valid) / sum(weight for weight, _ in valid)
                item['vocal_evidence_class'] = (max(item['vocal_class_scores'], key=item['vocal_class_scores'].get)
                                                if item['vocal_class_scores'] else None)
            energy = [(weight, 10 ** (float(w['rms_dbfs']) / 10)) for weight, w in weighted if w.get('rms_dbfs') is not None]
            if energy:
                item['rms_dbfs'] = 10 * math.log10(max(1e-24, sum(weight * value for weight, value in energy) / sum(weight for weight, value in energy)))
            else:
                item['rms_dbfs'] = None
        if changed:
            item.update(label=label, manual_override=True, needs_review=False)
        # The new split does not change the content label or review flags outside
        # the edited range. Every replacement remains traceable to this parent.
        item['edited_parent_bounds'] = [first / RATE, last / RATE]
        note = 'Range boundaries edited manually; automatic scores refer to the original inference windows.'
        item['note'] = (str(item.get('note') or '') + ' ' + note).strip()
        return item

    result = []
    if a > first:
        result.append(piece(first, a, False))
    selected = len(result)
    result.append(piece(a, b, True))
    if b < last:
        result.append(piece(b, last, False))
    return result, selected
