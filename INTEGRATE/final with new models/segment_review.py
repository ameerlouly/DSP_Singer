"""Track human decisions about proposed clips and export approved audio with its source provenance.

Function guide
--------------
utc_now, inspect_wav, check_source, load_session, audit
    Timestamp edits, examine WAV metadata and verify a session against the selected
    source.
validate_bounds, set_segment, candidate_trim_bounds, candidate_source_spans,
set_candidate_trim
    Validate time edits and map local trims to source audio.
pdf_audio_mapping, pdf_review_signature, require_pdf_review
    Check continuous source audio and bind a boundary review to that exact selection.
approve_segment, approve_segments, exclude_segment
    Record individual or batch approval and exclusion decisions.
unresolved_bounds, visible_unresolved, downstream_unresolved
    Find unreviewed spans and subtract audio that is already approved.
label_unresolved, mark_unresolved_reviewed, add_from_unresolved,
add_resegmented_candidates
    Review a span and create candidates while tracking which audio was used.
safe_candidate_path, extract_wav, item_source_ranges
    Resolve candidate files safely and read the designated audio intervals.
overlapping_approved, resolve_approved_overlaps
    Detect competing approved ranges and reconcile them with an audit entry.
report_html, approved_playback_queue, file_sha256, build_step5_manifest, playlist_html,
save_and_export
    Write the reviewed clip set, playback list, source mappings and integrity hashes.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import html
import hashlib
import json
import os
import shutil
import tempfile
import wave
import uuid

LABELS = ('speech', 'mixed', 'music', 'silence')
RATE = 16000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_wav(path):
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (RATE, 1, 2, 'NONE'):
            raise ValueError('Select the original 16 kHz, mono, 16-bit PCM WAV.')
        return w.getnframes(), w.getframerate()


def check_source(path, name, size, duration):
    path = Path(path)
    if not path.is_file() or path.name != name or path.stat().st_size != size:
        raise ValueError('Selected WAV filename or file size does not match the Step 4 review.')
    frames, rate = inspect_wav(path)
    if abs(frames/rate - float(duration)) > 1/rate:
        raise ValueError('Selected WAV duration does not match the Step 4 review.')
    return frames/rate


def load_session(input_file, source_wav):
    input_file, source_wav = Path(input_file).resolve(), Path(source_wav).resolve()
    payload = json.loads(input_file.read_text(encoding='utf-8-sig'))
    if payload.get('review_format') == 'step4_review_v1':
        data = payload
        root = Path(data['candidate_root']).resolve()
        source = data['source']
    else:
        root = input_file.parent
        source = {'audio_name':payload['audio_name'], 'audio_size':payload['audio_size'],
                  'duration_seconds':payload['duration_seconds']}
        data = {'review_format':'step4_review_v1', 'source':source,
                'candidate_root':str(root), 'original_metadata_file':str(input_file),
                'created_utc':utc_now(), 'changes':[], 'segments':[], 'unresolved':[],
                'pause_intervals':copy.deepcopy(payload.get('pause_intervals',[]))}
        for i, old in enumerate(payload.get('segments', []),1):
            data['segments'].append({'id':f'auto-{i:05d}', 'original':copy.deepcopy(old),
                'label':old['label'], 'status':'pending', 'source_start':float(old['start']),
                'source_end':float(old['end']), 'source_mode':'original_candidate',
                'candidate_file':old['file'], 'reviewed_by':None})
        for i, old in enumerate(payload.get('unresolved', []),1):
            data['unresolved'].append({'id':f'unresolved-{i:05d}', 'original':copy.deepcopy(old),
                'label':'unclassified', 'status':'pending', 'category':'unresolved', 'reviewed_by':None})
        for i, old in enumerate(payload.get('not_in_clips_or_unresolved', []),1):
            data['unresolved'].append({'id':f'gap-{i:05d}', 'original':copy.deepcopy(old),
                'label':'unclassified', 'status':'pending', 'category':'uncovered_gap', 'reviewed_by':None})
    check_source(source_wav, source['audio_name'], source['audio_size'], source['duration_seconds'])
    if not root.is_dir():
        raise ValueError('The original Step 4 output folder is no longer available.')
    data['candidate_root'] = str(root)
    return data


def audit(data, action, person, item_id, before, after):
    person=person.strip()
    if not person:
        raise ValueError('Enter the reviewer name first.')
    data['changes'].append({'time_utc':utc_now(), 'reviewer':person, 'action':action,
                            'item_id':item_id, 'before':copy.deepcopy(before), 'after':copy.deepcopy(after)})


def validate_bounds(start, end, duration):
    start, end = float(start), float(end)
    if not 0 <= start < end <= duration:
        raise ValueError('Start and end must be within the original audio and start must be before end.')
    frames = round(end*RATE)-round(start*RATE)
    if not 5*RATE <= frames <= 15*RATE:
        raise ValueError('Reviewed clips must be between 5 and 15 seconds.')
    return round(start*RATE)/RATE, round(end*RATE)/RATE


def set_segment(data, index, label, start, end, reviewer):
    if label not in LABELS:
        raise ValueError('Choose a valid label.')
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['segments'][index]
    before=copy.deepcopy(item)
    if abs(float(start)-item['source_start'])>1e-8 or abs(float(end)-item['source_end'])>1e-8:
        start,end=validate_bounds(start,end,data['source']['duration_seconds'])
        item.update(source_start=start,source_end=end,source_mode='continuous_source')
    item.update(label=label,status='pending',reviewed_by=None)
    audit(data,'edit_segment',reviewer,item['id'],before,item)



def candidate_trim_bounds(item):
    """Clip-local range in the original candidate WAV (not source times)."""
    if not item.get('original') or not item.get('candidate_file'):
        raise ValueError('Clip trimming is available for generated candidates. For manual clips, edit source times.')
    duration=float(item['original']['duration'])
    return (float(item.get('clip_trim_start',0.0)),float(item.get('clip_trim_end',duration)))


def candidate_source_spans(item):
    """Map a candidate trim back to the original audio without flattening joins."""
    trim_start,trim_end=candidate_trim_bounds(item)
    result=[]
    original=item['original']
    spans=original.get('source_spans',[])
    if not spans:
        source_start,source_end=float(original['start']),float(original['end'])
        if abs((source_end-source_start)-float(original['duration'])) > 1/RATE:
            raise ValueError('This older candidate lacks source mapping for safe trimming.')
        spans=[{'start':source_start,'end':source_end,'clip_start':0.0,'clip_end':float(original['duration'])}]
    if any('clip_start' not in span or 'clip_end' not in span for span in spans):
        total=sum(float(span['end'])-float(span['start']) for span in spans)
        if abs(total-float(original['duration']))>1/RATE:
            raise ValueError('This older candidate lacks source mapping for safe trimming.')
        offset=0.0
        mapped=[]
        for span in spans:
            length=float(span['end'])-float(span['start'])
            mapped.append(dict(span,clip_start=offset,clip_end=offset+length))
            offset+=length
        spans=mapped
    for span in spans:
        c0,c1=float(span['clip_start']),float(span['clip_end'])
        left,right=max(c0,trim_start),min(c1,trim_end)
        if right>left:
            result.append({'start':float(span['start'])+(left-c0),
                           'end':float(span['start'])+(right-c0),
                           'clip_start':left-trim_start,'clip_end':right-trim_start})
    return result


def set_candidate_trim(data,index,start,end,reviewer):
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['segments'][index]
    if item['source_mode']=='continuous_source':
        raise ValueError('This clip is a continuous source edit. Adjust source start/end instead.')
    candidate_trim_bounds(item)
    start,end=float(start),float(end)
    duration=float(item['original']['duration'])
    if not 0<=start<end<=duration:
        raise ValueError('Clip trim times must be inside the generated clip.')
    start,end=round(start*RATE)/RATE,round(end*RATE)/RATE
    if not 5*RATE <= round(end*RATE)-round(start*RATE) <= 15*RATE:
        raise ValueError('The reviewed clip must be 5–15 seconds long.')
    before=copy.deepcopy(item)
    draft=copy.deepcopy(item)
    draft['clip_trim_start'],draft['clip_trim_end']=start,end
    draft['source_mode']='original_candidate' if start==0 and abs(end-duration)<1/RATE else 'trimmed_candidate'
    spans=candidate_source_spans(draft)
    if not spans:
        raise ValueError('The selected range does not contain mapped source audio.')
    draft['source_start'],draft['source_end']=spans[0]['start'],spans[-1]['end']
    draft.update(status='pending',reviewed_by=None)
    item.clear();item.update(draft)
    audit(data,'trim_candidate',reviewer,item['id'],before,item)


def pdf_audio_mapping(data, item):
    """Validate PDF source continuity; artificial samples cannot become silence evidence."""
    if item['source_mode']=='continuous_source':
        a,b=validate_bounds(item['source_start'],item['source_end'],data['source']['duration_seconds'])
        return a,b
    left,right=candidate_trim_bounds(item)
    if not 5*RATE<=round(right*RATE)-round(left*RATE)<=15*RATE:
        raise ValueError('PDF Step 4 requires 5–15 seconds including natural edge context.')
    spans=candidate_source_spans(item)
    if not spans:raise ValueError('Missing original-source mapping; select a continuous source interval.')
    clip_cursor=0
    source_cursor=round(spans[0]['start']*RATE)
    first=source_cursor
    for span in spans:
        a,b,c,d=[round(float(span[k])*RATE) for k in ('start','end','clip_start','clip_end')]
        if a!=source_cursor or c!=clip_cursor or b-a!=d-c or b<=a:
            raise ValueError('PDF Step 4 requires continuous original audio: joined gaps or synthetic silence remain. Choose Original source times and review both edges.')
        source_cursor,clip_cursor=b,d
    if clip_cursor!=round(right*RATE)-round(left*RATE):
        raise ValueError('Synthetic edge padding remains. Select a continuous source interval with natural pauses before approval.')
    return validate_bounds(first/RATE,source_cursor/RATE,data['source']['duration_seconds'])


def pdf_review_signature(data,item):
    a,b=pdf_audio_mapping(data,item)
    return dict(source_start=a,source_end=b,label=item['label'],source_mode=item['source_mode'])


def require_pdf_review(data,item):
    signature=pdf_review_signature(data,item)
    checked=item.get('pdf_boundary_review') or {}
    if checked.get('signature')!=signature or not checked.get('reviewer'):
        raise ValueError('Reapprove this clip after listening: confirm complete words and 0.1–0.2 s natural pause at both edges.')
    return signature['source_start'],signature['source_end']


def approve_segment(data, index, reviewer):
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['segments'][index]
    before=copy.deepcopy(item)
    if item['source_mode']=='continuous_source':
        validate_bounds(item['source_start'],item['source_end'],data['source']['duration_seconds'])
    else:
        trim_start,trim_end=candidate_trim_bounds(item)
        duration=trim_end-trim_start
        if not 5 <= duration <= 15:
            raise ValueError('This candidate is not between 5 and 15 seconds; correct its boundaries first.')
    signature=pdf_review_signature(data,item)
    item['pdf_boundary_review']=dict(reviewer=reviewer,time_utc=utc_now(),signature=signature,
        attestation='Annotator confirms complete words, real pauses and about 0.1–0.2 s natural pause at both edges; no synthetic padding or joins.')
    item.update(status='approved', reviewed_by=reviewer)
    audit(data,'approve_segment',reviewer,item['id'],before,item)


def approve_segments(data, indices, reviewer):
    """Approve selected clips atomically, retaining an audit entry per clip."""
    if not reviewer.strip():
        raise ValueError('Enter the reviewer name first.')
    indices = list(dict.fromkeys(indices))
    if not indices:
        raise ValueError('Select at least one clip to approve.')
    draft = copy.deepcopy(data)
    count = 0
    for index in indices:
        if not isinstance(index, int) or not 0 <= index < len(draft['segments']):
            raise ValueError('A selected clip no longer exists.')
        item = draft['segments'][index]
        if item['status'] == 'excluded':
            raise ValueError('An excluded clip is selected. Edit it individually to return it to pending before approval.')
        checked=(item.get('pdf_boundary_review') or {}).get('signature')
        if item['status'] != 'approved' or checked!=pdf_review_signature(draft,item):
            try:
                approve_segment(draft, index, reviewer)
            except ValueError as exc:
                raise ValueError(f"Clip {item['id']}: {exc}") from exc
            count += 1
    data['segments'] = draft['segments']
    data['changes'] = draft['changes']
    return count



def exclude_segment(data,index,reviewer):
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['segments'][index]
    before=copy.deepcopy(item)
    item.update(status='excluded',reviewed_by=reviewer)
    audit(data,'exclude_segment',reviewer,item['id'],before,item)


def unresolved_bounds(item):
    """Current bounds; older saved sessions only have original bounds."""
    return (float(item.get('source_start', item['original']['start'])),
            float(item.get('source_end', item['original']['end'])))


def visible_unresolved(data):
    """Show only source ranges outside active manually created clips."""
    covered = sorted((round(float(item['source_start']) * RATE),
                      round(float(item['source_end']) * RATE))
                     for item in data.get('segments', [])
                     if item.get('parent_unresolved_id') and item.get('original') is None
                     and item.get('status') != 'excluded')
    result = []
    for index, item in enumerate(data.get('unresolved', [])):
        first, last = (round(t * RATE) for t in unresolved_bounds(item))
        cursor = first
        for left, right in covered:
            if left >= last:
                break
            if right <= cursor:
                continue
            if left > cursor:
                result.append((index, item, cursor / RATE, min(left, last) / RATE))
            cursor = max(cursor, right)
            if cursor >= last:
                break
        if cursor < last:
            result.append((index, item, cursor / RATE, last / RATE))
    return result


def downstream_unresolved(data, approved_ranges):
    """Yield the source samples still outside approved clips, preserving parent IDs."""
    pending_manual = [(float(item['source_start']), float(item['source_end']))
                      for item in data.get('segments', [])
                      if item.get('parent_unresolved_id') and item.get('original') is None
                      and item.get('status') == 'pending']
    covered = sorted((round(a*RATE), round(b*RATE)) for a,b in [*approved_ranges, *pending_manual])
    total = round(data['source']['duration_seconds']*RATE)
    for item in data.get('unresolved', []):
        label = item.get('label') or 'unclassified'
        if item.get('category') == 'uncovered_gap' and label not in ('speech','mixed'):
            continue
        first, last = (round(t*RATE) for t in unresolved_bounds(item))
        if not 0 <= first < last <= total:
            raise ValueError(f"Invalid source range for unresolved span {item['id']}.")
        cursor = first
        for left, right in covered:
            if left >= last:
                break
            if right <= cursor:
                continue
            if left > cursor:
                yield item, cursor/RATE, left/RATE
            cursor = max(cursor, right)
        if cursor < last:
            yield item, cursor/RATE, last/RATE


def label_unresolved(data,index,label,reviewer,start=None,end=None):
    if label not in LABELS:
        raise ValueError('Choose a valid label.')
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['unresolved'][index]
    old_start,old_end=unresolved_bounds(item)
    start=old_start if start is None else float(start)
    end=old_end if end is None else float(end)
    if not old_start-1e-9 <= start < end <= old_end+1e-9:
        raise ValueError('The edited range must be inside this unresolved span; start must be before end.')
    # Quantize to the source sample grid. Avoid invisible microscopic leftovers.
    start=round(start*RATE)/RATE
    end=round(end*RATE)/RATE
    if start < old_start-1/RATE or end > old_end+1/RATE or start >= end:
        raise ValueError('The edited range is invalid after sample rounding.')
    before=copy.deepcopy(item)
    leftovers=[]
    for a,b in ((old_start,start),(end,old_end)):
        if round(b*RATE)-round(a*RATE) >= 1:
            sibling={'id':'split-'+uuid.uuid4().hex[:12], 'original':copy.deepcopy(item['original']),
                     'source_start':a, 'source_end':b, 'label':item['label'] if item['label'] in LABELS else 'unclassified', 'status':'pending',
                     'category':item.get('category','unresolved'), 'reviewed_by':None,
                     'split_from':item['id']}
            leftovers.append(sibling)
    item.update(label=label,source_start=start,source_end=end,status='pending',reviewed_by=None)
    data['unresolved'].extend(leftovers)
    audit(data,'label_or_trim_unresolved',reviewer,item['id'],before,
          {'selected':item,'preserved_leftovers':leftovers})


def mark_unresolved_reviewed(data,index,reviewer):
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    item=data['unresolved'][index]
    if item['label'] not in LABELS:
        raise ValueError('Set the label before marking the span reviewed.')
    before=copy.deepcopy(item)
    item.update(status='reviewed',reviewed_by=reviewer)
    audit(data,'mark_unresolved_reviewed',reviewer,item['id'],before,item)


def add_from_unresolved(data,index,start,end,label,reviewer):
    if label not in LABELS:
        raise ValueError('Choose a valid label.')
    reviewer=reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    start,end=validate_bounds(start,end,data['source']['duration_seconds'])
    parent=data['unresolved'][index]
    selected_start, selected_end = unresolved_bounds(parent)
    if start < selected_start - 1/RATE or end > selected_end + 1/RATE:
        raise ValueError('The new clip must be inside the selected unresolved span or gap.')
    covered = [(a, b) for i, _, a, b in visible_unresolved(data) if i == index]
    if not any(a - 1/RATE <= start < end <= b + 1/RATE for a, b in covered):
        raise ValueError('The new clip overlaps a segment already added from an unresolved span.')
    item={'id':'manual-'+uuid.uuid4().hex[:12], 'original':None,
          'label':label, 'status':'pending', 'source_start':start, 'source_end':end,
          'source_mode':'continuous_source', 'candidate_file':None,
          'parent_unresolved_id':parent['id'], 'reviewed_by':None}
    data['segments'].append(item)
    audit(data,'add_clip_from_unresolved',reviewer,item['id'],None,item)
    return len(data['segments'])-1



def add_resegmented_candidates(data,index,candidates,reviewer):
    """Register generated drafts while keeping the unresolved source span pending."""
    reviewer=reviewer.strip()
    if not reviewer:raise ValueError('Enter the reviewer name first.')
    if not candidates:raise ValueError('No candidates were generated.')
    parent=data['unresolved'][index]
    start,end=unresolved_bounds(parent)
    if parent['label'] not in ('speech','mixed'):
        raise ValueError('Apply the speech or mixed label to this range first.')
    cursor=round(start*RATE)
    prepared=[]
    for old in candidates:
        a,b=round(float(old['start'])*RATE),round(float(old['end'])*RATE)
        if a!=cursor or b<=a or old['label']!=parent['label']:
            raise ValueError('Generated candidates do not form the selected source interval.')
        path=safe_candidate_path(data['candidate_root'],old['file'])
        frames,rate=inspect_wav(path)
        if not 5*rate<=frames<=15*rate:
            raise ValueError('Generated candidate duration is not between 5 and 15 seconds.')
        prepared.append({'id':'resegmented-'+uuid.uuid4().hex[:12],'original':copy.deepcopy(old),
            'label':old['label'],'status':'pending','source_start':a/RATE,'source_end':b/RATE,
            'source_mode':'original_candidate','candidate_file':old['file'],
            'parent_unresolved_id':parent['id'],'reviewed_by':None})
        cursor=b
    if cursor!=round(end*RATE):
        raise ValueError('Generated candidates do not cover the end of the selected interval.')
    before=copy.deepcopy(parent)
    first=len(data['segments'])
    data['segments'].extend(prepared)
    parent.setdefault('generated_candidate_ids',[]).extend(x['id'] for x in prepared)
    parent['status']='pending'
    parent['reviewed_by']=None
    audit(data,'auto_resegment_unresolved',reviewer,parent['id'],before,
          {'unresolved':parent,'generated_candidate_ids':[x['id'] for x in prepared]})
    return first


def safe_candidate_path(root, name):
    root=Path(root).resolve()
    target=(root/name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError('A candidate file path is outside the Step 4 folder.') from exc
    if not target.is_file():
        raise FileNotFoundError(f'Candidate WAV is missing: {name}')
    return target


def extract_wav(source, dest, start, end):
    with wave.open(str(source),'rb') as src:
        start_frame,end_frame=round(start*RATE),round(end*RATE)
        if end_frame>src.getnframes():
            raise ValueError('Selected interval exceeds the source audio.')
        src.setpos(start_frame)
        with wave.open(str(dest),'wb') as dst:
            dst.setnchannels(1);dst.setsampwidth(2);dst.setframerate(RATE)
            remaining=end_frame-start_frame
            while remaining:
                count=min(remaining, RATE*10)
                data=src.readframes(count)
                if len(data)!=count*2:
                    raise IOError('Source WAV ended unexpectedly.')
                dst.writeframesraw(data)
                remaining-=count



def item_source_ranges(item):
    if item['source_mode'] == 'continuous_source':
        return [(item['source_start'], item['source_end'])]
    try:
        return [(x['start'],x['end']) for x in candidate_source_spans(item)]
    except (ValueError,KeyError):
        # Conservative envelope for older metadata with no exact piece map.
        return [(float(item['source_start']),float(item['source_end']))]


def overlapping_approved(data):
    approved=[x for x in data['segments'] if x['status']=='approved']
    results=[]
    for i,left in enumerate(approved):
        for right in approved[i+1:]:
            seconds=sum(max(0.0,min(b,d)-max(a,c)) for a,b in item_source_ranges(left) for c,d in item_source_ranges(right))
            if seconds > 1/RATE:
                results.append({'left':left['id'],'right':right['id'],'seconds':seconds})
    return results


def resolve_approved_overlaps(data, reviewer):
    """Remove fully duplicated clips and trim only duplicated clip edges.

    All calculations use source and clip sample positions. Internal overlaps,
    missing mappings, and trims outside 5-15 seconds are left for review.
    """
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError('Enter the reviewer name first.')
    draft = copy.deepcopy(data)
    covered = []
    trimmed = excluded = 0
    warnings = []
    ordered = sorted((x for x in draft['segments'] if x['status'] == 'approved'),
                     key=lambda x: x['source_start'])
    for item in ordered:
        try:
            if item['source_mode'] == 'continuous_source':
                start = round(item['source_start'] * RATE)
                end = round(item['source_end'] * RATE)
                spans = [{'start':start / RATE,'end':end / RATE,
                          'clip_start':0,'clip_end':(end-start)/RATE}]
                length = end-start
            else:
                spans = candidate_source_spans(item)
                candidate_start,candidate_end = candidate_trim_bounds(item)
                length = round(candidate_end*RATE)-round(candidate_start*RATE)
            if not spans:
                raise ValueError('no source mapping')
            mappings=[]
            for span in spans:
                a,b,c,d=(round(float(span[k])*RATE) for k in ('start','end','clip_start','clip_end'))
                if b<=a or d<=c or abs((b-a)-(d-c))>1 or c<0 or d>length:
                    raise ValueError('invalid source mapping')
                mappings.append((a,b,c,d))
            duplicate=[]
            for a,b,c,d in mappings:
                for x,y in covered:
                    left,right=max(a,x),min(b,y)
                    if left<right: duplicate.append((c+left-a,c+right-a))
            if not duplicate:
                covered.extend((a,b) for a,b,_,_ in mappings)
                continue
            unique=[]
            for a,b,c,d in mappings:
                parts=[(a,b)]
                for x,y in covered:
                    parts=[piece for p,q in parts for piece in ((p,min(q,x)),(max(p,y),q)) if piece[0]<piece[1]]
                unique.extend((c+p-a,c+q-a) for p,q in parts)
            before=copy.deepcopy(item)
            if not unique:
                item.update(status='excluded',reviewed_by=reviewer)
                audit(draft,'auto_exclude_duplicate',reviewer,item['id'],before,item)
                excluded+=1
                continue
            left=min(p for p,q in unique); right=max(q for p,q in unique)
            if any(max(p,left)<min(q,right) for p,q in duplicate):
                raise ValueError('overlap is inside the clip, not just at its edges')
            # Keep unduplicated artificial edge padding if present.
            first=min(c for _,_,c,_ in mappings); last=max(d for _,_,_,d in mappings)
            new_left=0 if left==first else left
            new_right=length if right==last else right
            if not 5*RATE <= new_right-new_left <= 15*RATE:
                raise ValueError('removing the overlap would make the clip shorter than 5 seconds')
            if item['source_mode']=='continuous_source':
                item['source_start']=(start+new_left)/RATE
                item['source_end']=(start+new_right)/RATE
            else:
                item['clip_trim_start']=(round(candidate_start*RATE)+new_left)/RATE
                item['clip_trim_end']=(round(candidate_start*RATE)+new_right)/RATE
                item['source_mode']='trimmed_candidate'
                updated=candidate_source_spans(item)
                item['source_start'],item['source_end']=updated[0]['start'],updated[-1]['end']
            item['overlap_cleanup_note']='Automatically trimmed duplicate audio; boundary has not been independently verified.'
            audit(draft,'auto_trim_overlap',reviewer,item['id'],before,item)
            trimmed+=1
            covered.extend((round(x*RATE),round(y*RATE)) for x,y in item_source_ranges(item))
        except (KeyError,TypeError,ValueError) as exc:
            warnings.append(f"{item.get('id','clip')}: {exc}")
            # Preserve conservative source coverage so later clips cannot silently duplicate it.
            covered.extend((round(x*RATE),round(y*RATE)) for x,y in item_source_ranges(item))
    data['segments']=draft['segments']; data['changes']=draft['changes']
    return {'trimmed':trimmed,'excluded':excluded,'warnings':warnings,
            'remaining':overlapping_approved(data)}


def report_html(data, exports):
    e=lambda x:html.escape(str(x),quote=True)
    rows=[]
    for item in data['segments']:
        filename=exports.get(item['id'],'')
        audio=f"<audio controls preload='none' src='{e(filename)}'></audio>" if filename else ''
        orig=item.get('original') or {}
        if item['source_mode'] in ('original_candidate','trimmed_candidate'):
            try:spans=candidate_source_spans(item)
            except (ValueError,KeyError):spans=[]
        else:
            spans=[{'start':item['source_start'],'end':item['source_end']}]
        piece_text=', '.join(f"{span['start']:.3f}–{span['end']:.3f}" for span in spans)
        method=orig.get('kind','manual') if item['source_mode'] in ('original_candidate','trimmed_candidate') else 'manual source range'
        notes='; '.join(orig.get('notes',[])) if item['source_mode'] in ('original_candidate','trimmed_candidate') else 'Source range selected by reviewer.'
        if item.get('overlap_cleanup_note'):
            notes += '; ' + item['overlap_cleanup_note']
        rows.append(f"<tr><td>{e(item['id'])}</td><td>{item['source_start']:.3f}</td><td>{item['source_end']:.3f}</td>"
                    f"<td>{e(item['label'])}</td><td>{e(item['status'])}</td><td>{e(item['source_mode'])}</td>"
                    f"<td>{e(item.get('reviewed_by') or '')}</td><td>{e(method)}</td><td>{e(piece_text)}</td><td>{e(notes)}</td><td>{e(filename)}</td><td>{audio}</td></tr>")
    unresolved=[]
    for _, item, current_start, current_end in visible_unresolved(data):
        o=item['original']
        unresolved.append(f"<tr><td>{e(item['id'])}</td><td>{current_start:.3f}</td><td>{current_end:.3f}</td>"
                          f"<td>{e(item['label'])}</td><td>{e(item['status'])}</td><td>{e(o.get('reason',''))}</td><td>{len(item.get('generated_candidate_ids',[]))}</td></tr>")
    changes=''.join(f"<li>{e(c['time_utc'])} — {e(c['reviewer'])}: {e(c['action'])} ({e(c['item_id'])})</li>" for c in data['changes'])
    overlaps=overlapping_approved(data)
    overlap_rows=''.join(f"<li>{e(x['left'])} and {e(x['right'])}: {x['seconds']:.3f} s source overlap</li>" for x in overlaps)
    approved=sum(s['status']=='approved' for s in data['segments'])
    pending=sum(s['status']=='pending' for s in data['segments'])
    excluded=sum(s['status']=='excluded' for s in data['segments'])
    unresolved_pending=sum(s['status']!='reviewed' for _,s,_,_ in visible_unresolved(data))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step 4 review</title><style>body{{font:15px/1.5 Segoe UI,Arial;background:#f2f6f7;color:#173a40;margin:0}}header{{background:#173a40;color:white;padding:24px}}main{{padding:18px;max-width:1400px;margin:auto}}section{{background:white;border:1px solid #dbe5e6;border-radius:10px;padding:18px;margin-bottom:16px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{text-align:left;border-bottom:1px solid #ddd;padding:8px}}th{{background:#eaf4f4}}audio{{width:180px}}.warning{{background:#fff4d9}}</style></head><body><header><h1>Step 4 review</h1><p>{e(data['source']['audio_name'])}</p></header><main>
<section><h2>Summary</h2><p>Approved clips: {approved}. Pending clips: {pending}. Excluded clips: {excluded}. Unresolved spans still pending: {unresolved_pending}. Only approved clips are exported.</p><p>Original audio and automatic candidates are preserved. Each item records who reviewed it and the full audit is in review_session.json.</p></section>
<section class="warning"><b>Boundary review:</b> approving means the annotator has checked the cut. Some original candidates contain synthetic silence or multiple source ranges. An unchanged candidate retains these; the source start/end columns are its outer envelope, not proof of a continuous interval.</section>
<section><h2>Segments</h2><div class="scroll"><table><tr><th>ID</th><th>Source start</th><th>Source end</th><th>Label</th><th>Status</th><th>Mode</th><th>Reviewer</th><th>Method</th><th>Original source pieces (s)</th><th>Notes</th><th>Export</th><th>Play</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Overlapping approved clips</h2><p>{len(overlaps)} pair(s) overlap in original source time. This can duplicate audio; verify each pair.</p><ul>{overlap_rows}</ul></section>
<section><h2>Unresolved spans</h2><p>Marking a span reviewed is an annotator assertion and does not automatically create clips or prove full coverage.</p><div class="scroll"><table><tr><th>ID</th><th>Start</th><th>End</th><th>Label</th><th>Status</th><th>Reason</th><th>New draft candidates</th></tr>{''.join(unresolved)}</table></div></section>
<section><h2>Change history</h2><ul>{changes}</ul></section></main></body></html>'''




def approved_playback_queue(data,source_wav):
    """Ordered (path, start, end, label, id) for separate approved WAVs."""
    source=Path(source_wav).resolve()
    info=data['source']
    check_source(source,info['audio_name'],info['audio_size'],info['duration_seconds'])
    result=[]
    for item in sorted((x for x in data['segments'] if x['status']=='approved'),key=lambda x:x['source_start']):
        require_pdf_review(data,item)
        if item['source_mode']=='continuous_source':
            a,b=validate_bounds(item['source_start'],item['source_end'],info['duration_seconds'])
            path=source
        else:
            path=safe_candidate_path(data['candidate_root'],item['candidate_file'])
            frames,rate=inspect_wav(path)
            a,b=candidate_trim_bounds(item)
            if not 0<=a<b<=frames/rate or not 5*rate<=round(b*rate)-round(a*rate)<=15*rate:
                raise ValueError(f"Invalid audio boundaries for {item['id']}.")
        result.append((str(path),a,b,item['label'],item['id']))
    return result


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def build_step5_manifest(data, ordered, exports, output):
    clips=[]
    for order,item in enumerate(ordered,1):
        filename=exports[item['id']]
        frames,rate=inspect_wav(Path(output)/filename)
        source_mode=item['source_mode']
        mapping_warning=None
        if source_mode=='continuous_source':
            spans=[{'start':item['source_start'],'end':item['source_end'],
                    'clip_start':0.0,'clip_end':(frames/rate)}]
        else:
            try:
                spans=candidate_source_spans(item)
            except (ValueError,KeyError):
                spans=[]
                mapping_warning='Original candidate did not provide complete source mapping.'
        original=item.get('original') or {}
        original_padding={'start_seconds':float(original.get('synthetic_silence_start_seconds',0.0)),
                          'end_seconds':float(original.get('synthetic_silence_end_seconds',0.0))}
        synthetic={'start_seconds':0.0,'end_seconds':0.0}
        if source_mode in ('original_candidate','trimmed_candidate'):
            a,b=candidate_trim_bounds(item)
            duration=float(original['duration'])
            synthetic['start_seconds']=max(0.0,min(b,original_padding['start_seconds'])-a)
            synthetic['end_seconds']=max(0.0,b-max(a,duration-original_padding['end_seconds']))
        clips.append({'order':order,'file':filename,'label':item['label'],'status':'approved',
            'pdf_boundary_review':copy.deepcopy(item.get('pdf_boundary_review')),
            'reviewer':item.get('reviewed_by'),'sample_rate_hz':rate,'sample_count':frames,
            'duration_seconds':frames/rate,'sha256':file_sha256(Path(output)/filename),
            'source_mode':source_mode,'source_spans':spans,'source_mapping_warning':mapping_warning,
            'synthetic_silence':synthetic,'speech_or_mixed':item['label'] in ('speech','mixed')})
    return {'format':'speech_corpus_step5_inputs_v1','created_utc':utc_now(),
            'source_audio':{'name':data['source']['audio_name'],'size_bytes':data['source']['audio_size'],
                            'duration_seconds':data['source']['duration_seconds']},
            'description':'Ordered reviewed WAV files, separate on disk. Paths are relative to this manifest.',
            'clips':clips}


def playlist_html(clips):
    # Filenames are generated locally from fixed labels and digits, not user HTML.
    tracks=json.dumps([{'file':x['file'],'label':x['label']} for x in clips],ensure_ascii=True)
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Reviewed clips playlist</title>
<style>body{font:16px/1.5 Segoe UI,Arial;max-width:900px;margin:30px auto;padding:0 18px;color:#173a40}
button{padding:8px 16px}audio{width:100%}li{margin:6px 0}button.track{border:0;background:none;color:#176d76;text-decoration:underline;cursor:pointer}</style>
</head><body><h1>Reviewed clips playlist</h1><p>The original clips remain separate files. Playback moves to the next clip automatically.</p>
<button id='play'>Play from first clip</button><p id='current'></p><audio id='audio' controls></audio><ol id='list'></ol>
<script>const tracks=TRACKS;let index=0;const audio=document.getElementById('audio');const current=document.getElementById('current');
function playAt(i){if(i<0||i>=tracks.length)return;index=i;audio.src=tracks[i].file;current.textContent=(i+1)+' / '+tracks.length+' — '+tracks[i].file;audio.play().catch(e=>{current.textContent+=' — click Play on the audio control';});}
const list=document.getElementById('list');tracks.forEach((t,i)=>{const li=document.createElement('li');const b=document.createElement('button');b.className='track';b.textContent=t.file+' ('+t.label+')';b.onclick=()=>playAt(i);li.appendChild(b);list.appendChild(li);});
if(!tracks.length){current.textContent='No approved clips were exported.';document.getElementById('play').disabled=true;}document.getElementById('play').onclick=()=>playAt(0);audio.onended=()=>{if(index+1<tracks.length)playAt(index+1);else current.textContent='Playlist finished.';};</script></body></html>""".replace('TRACKS',tracks)


def save_and_export(data, source_wav, destination):
    source=Path(source_wav).resolve()
    info=data['source']
    check_source(source,info['audio_name'],info['audio_size'],info['duration_seconds'])
    root=Path(data['candidate_root']).resolve()
    for item in data['segments']:
        if item['status']!='approved':
            continue
        if item['label'] not in LABELS:
            raise ValueError('Invalid approved label.')
        require_pdf_review(data,item)
        if item['source_mode'] in ('original_candidate','trimmed_candidate'):
            path=safe_candidate_path(root,item['candidate_file'])
            frames,rate=inspect_wav(path)
            start,end=candidate_trim_bounds(item)
            if not 0<=start<end<=frames/rate or not 5*rate<=round(end*rate)-round(start*rate)<=15*rate:
                raise ValueError(f"Candidate {item['id']} has invalid trim boundaries.")
        elif item['source_mode']=='continuous_source':
            validate_bounds(item['source_start'],item['source_end'],info['duration_seconds'])
        else:
            raise ValueError('Unknown clip source mode.')
    dest=Path(destination).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    # Never overwrite an earlier export, even if the user saves repeatedly.
    out=dest/('step4_review_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir()
    exports={}
    ordered=sorted((x for x in data['segments'] if x['status']=='approved'), key=lambda x:x['source_start'])
    for i,item in enumerate(ordered,1):
        name=f"{item['label']}_{i:03d}.wav"
        target=out/name
        source_start,source_end=require_pdf_review(data,item)
        extract_wav(source,target,source_start,source_end)
        exports[item['id']]=name
    saved=copy.deepcopy(data)
    saved['last_saved_utc']=utc_now()
    saved['last_exported_files']=exports
    (out/'review_session.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'region_review_report.html').write_text(report_html(saved,exports),encoding='utf-8')
    manifest=build_step5_manifest(saved,ordered,exports,out)
    (out/'step5_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'reviewed_playlist.m3u8').write_text('#EXTM3U\n'+''.join(item['file']+'\n' for item in manifest['clips']),encoding='utf-8')
    (out/'reviewed_playlist.html').write_text(playlist_html(manifest['clips']),encoding='utf-8')
    return {'output_dir':str(out),'exported':len(exports),'playlist':'reviewed_playlist.html','step5_manifest':'step5_manifest.json',
            'pending_segments':sum(x['status']=='pending' for x in data['segments']),
            'excluded_segments':sum(x['status']=='excluded' for x in data['segments']),
            'pending_unresolved':sum(x['status']!='reviewed' for _,x,_,_ in visible_unresolved(data)),
            'overlapping_pairs':len(overlapping_approved(data))}
