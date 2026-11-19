"""Read the separate approved WAVs prepared for later Step 5 processing."""
from pathlib import Path
import json
from step4_review import file_sha256, inspect_wav


def load_reviewed_clips(manifest_path, verify_hash=True):
    """Return ordered clip records with absolute audio_path values.

    Paths in the saved manifest are relative to the manifest folder. When
    verify_hash is true, changed WAV files are rejected. This function does not
    calculate SNR or run speech recognition.
    """
    manifest_path=Path(manifest_path).resolve()
    root=manifest_path.parent
    data=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if data.get('format')!='speech_corpus_step5_inputs_v1' or not isinstance(data.get('clips'),list):
        raise ValueError('Not a supported reviewed Step 5 manifest.')
    result=[]
    for expected,clip in enumerate(data['clips'],1):
        if clip.get('order')!=expected or clip.get('status')!='approved':
            raise ValueError('Manifest clip order or approval status is invalid.')
        name=clip['file']
        path=(root/name).resolve()
        try:path.relative_to(root)
        except ValueError as exc:raise ValueError('A clip path is outside the manifest folder.') from exc
        if not path.is_file():raise FileNotFoundError(f'Reviewed clip not found: {name}')
        frames,rate=inspect_wav(path)
        if frames!=clip['sample_count'] or rate!=clip['sample_rate_hz']:
            raise ValueError(f'Audio format or length changed for {name}.')
        if verify_hash and file_sha256(path)!=clip['sha256']:
            raise ValueError(f'Audio contents changed for {name}.')
        entry=dict(clip)
        entry['audio_path']=str(path)
        result.append(entry)
    return result
