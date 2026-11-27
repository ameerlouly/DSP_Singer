"""User-visible edit history: unresolved cuts and human transcript choices only.

Approval evidence, quality measurements and inclusion preferences remain in their
own state fields. They do not belong in the per-clip activity log.
"""
CUT_ACTIONS = {'add_clip_from_unresolved', 'auto_resegment_unresolved'}
TRANSCRIPT_ACTIONS = {'transcript_review', 'choose_asr_model', 'edit_final_segment'}


def keep_event(event):
    if not isinstance(event, dict):
        return False
    action = event.get('action')
    if action in CUT_ACTIONS:
        return True
    if action == 'label_or_trim_unresolved':
        before = event.get('before') or {}
        after = event.get('after') or {}
        selected = after.get('selected') or after
        if after.get('preserved_leftovers'):
            return True
        def bounds(item):
            original = item.get('original') or {}
            return (item.get('source_start', original.get('start')), item.get('source_end', original.get('end')))
        return bounds(before) != bounds(selected)
    if action == 'edit_final_segment':
        before, after = event.get('before'), event.get('after')
        return isinstance(before, dict) and isinstance(after, dict) and before.get('final_transcript') != after.get('final_transcript')
    # Legacy transcript_review events represented explicit human decisions;
    # retain them since the old format did not distinguish button clicks.
    return action in TRANSCRIPT_ACTIONS


def events_only(events):
    return [event for event in (events or []) if keep_event(event)]


def segment_session_view(session):
    """Filter exported history without modifying the live session/signature."""
    if session is None:
        return None
    return dict(session, changes=events_only(session.get('changes')))


def record_view(record):
    return dict(record, human_changes=events_only(record.get('human_changes')))


def source_review_view(source):
    if source is None:
        return None
    result = dict(source)
    if 'step4' in result:
        result['step4'] = segment_session_view(result['step4'])
    if 'region_changes' in result:
        result['region_changes'] = [e for e in (result['region_changes'] or []) if e.get('action') != 'auto_approve_region']
    return result
