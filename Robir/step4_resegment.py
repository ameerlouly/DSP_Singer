"""Re-segment one user-labeled unresolved source interval for manual review."""
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from step4_cut_segments import detect_speech, refine_long_islands
from step4_fallback import RATE, PAD, MIN_CONTENT, MAX_CONTENT, low_energy_boundary


def plan_splits(audio, first, last, speech_islands):
    """Partition every source sample, preferring VAD gaps over low-energy cuts.

    Returns a list of review-only ranges. Artificial edge silence is accounted
    for in the 5–15 second output length; it is not evidence of a real pause.
    """
    if last-first<MIN_CONTENT:
        raise ValueError('The selected audio is too short for a 5-second candidate. Choose more source context.')
    pauses=[]
    for left,right in zip(speech_islands,speech_islands[1:]):
        if right[0]-left[1]>=.079:
            pauses.append(round((left[1]+right[0])*RATE/2))
    boundaries=[first]
    kinds=[]
    cursor=first
    while last-cursor>MAX_CONTENT:
        low=cursor+MIN_CONTENT
        high=min(cursor+MAX_CONTENT,last-MIN_CONTENT)
        if high<low:
            raise ValueError('Cannot form 5–15 second candidates from this range.')
        desired=min(high,max(low,cursor+10*RATE))
        choices=[point for point in pauses if low<=point<=high]
        if choices:
            cut=min(choices,key=lambda point:abs(point-desired))
            kind='vad_pause_candidate'
        else:
            cut,_=low_energy_boundary(audio,low,high)
            kind='low_energy_fallback'
        if not low<=cut<=high:
            raise RuntimeError('Invalid candidate boundary.')
        boundaries.append(cut);kinds.append(kind);cursor=cut
    boundaries.append(last)
    result=[]
    for i,(a,b) in enumerate(zip(boundaries,boundaries[1:])):
        left='selected_range_start' if i==0 else kinds[i-1]
        right='selected_range_end' if i==len(kinds) else kinds[i]
        result.append({'first':a,'last':b,'left_boundary_method':left,'right_boundary_method':right})
    return result


def resegment_range(source_wav, candidate_root, start, end, label, progress=lambda s:None):
    """Create new draft WAVs beneath candidate_root. Never modify the source.

    Each candidate gets 0.1 s artificial silence at each edge. Returned source
    spans partition [start,end] exactly on the 16-kHz sample grid. Everything
    remains pending review, including VAD-proposed cuts.
    """
    if label not in ('speech','mixed'):
        raise ValueError('Label this range speech or mixed before auto-segmenting it.')
    start,end=float(start),float(end)
    first,last=round(start*RATE),round(end*RATE)
    if first<0 or first>=last:
        raise ValueError('Choose a valid source range.')
    with sf.SoundFile(source_wav) as audio:
        if (audio.samplerate,audio.channels,audio.subtype)!=(RATE,1,'PCM_16'):
            raise ValueError('The source WAV must be 16 kHz, mono, 16-bit PCM.')
        if last>len(audio):
            raise ValueError('The selected range exceeds the source WAV.')
        if last-first<MIN_CONTENT:
            raise ValueError('The selected audio is too short for a 5-second candidate. Choose more source context.')
        from silero_vad import load_silero_vad
        import torch
        torch.set_num_threads(1)
        progress('Loading Silero VAD for the selected range...')
        model=load_silero_vad()
        progress('Detecting possible pauses...')
        islands=detect_speech(audio,first/RATE,last/RATE,model,progress)
        if islands:
            islands=refine_long_islands(audio,islands,model,progress)
        else:
            progress('No VAD speech detected; using low-energy candidates for review.')
        plans=plan_splits(audio,first,last,islands)
        root=Path(candidate_root).resolve()
        if not root.is_dir():
            raise ValueError('The Step 4 candidate folder is unavailable.')
        draft=Path(tempfile.mkdtemp(prefix='resegmented_',dir=root))
        records=[]
        pad=np.zeros((PAD,1),dtype='float32')
        for i,part in enumerate(plans,1):
            progress(f'Preparing draft clip {i} of {len(plans)}...')
            a,b=part['first'],part['last']
            audio.seek(a)
            samples=audio.read(b-a,dtype='float32',always_2d=True)
            if len(samples)!=b-a:
                raise IOError(f'Source audio ended unexpectedly. Draft files, if any, are in {draft}')
            full=np.concatenate((pad,samples,pad),axis=0)
            if not 5*RATE<=len(full)<=15*RATE:
                raise RuntimeError('Generated candidate duration is out of range.')
            filename=f'candidate_{i:04d}.wav'
            sf.write(draft/filename,full,RATE,subtype='PCM_16')
            left,right=part['left_boundary_method'],part['right_boundary_method']
            low_energy='low_energy_fallback' in (left,right)
            notes=['Automatically re-segmented unresolved source; listen to both boundaries.',
                   'Artificial 0.1 s silence at each edge is not a detected natural pause.',
                   f'Boundary methods: start={left}, end={right}.']
            if low_energy:
                notes.append('A low-energy boundary is not a verified pause and may split a word.')
            records.append({'file':(draft/filename).relative_to(root).as_posix(),
                'kind':'low_energy_fallback' if low_energy else 'vad_pause_candidate',
                'start':a/RATE,'end':b/RATE,'duration':len(full)/RATE,'label':label,
                'source_spans':[{'start':a/RATE,'end':b/RATE,'clip_start':PAD/RATE,'clip_end':(PAD+b-a)/RATE}],
                'removed_gaps':[],'synthetic_silence_start_seconds':PAD/RATE,
                'synthetic_silence_end_seconds':PAD/RATE,'left_boundary_method':left,
                'right_boundary_method':right,'status':'needs_review','notes':notes})
    return records
