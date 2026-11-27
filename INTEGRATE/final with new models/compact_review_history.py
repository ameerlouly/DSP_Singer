"""Store large review histories by reference to the unchanged corpus audit.

Small histories stay byte-for-byte compatible with the previous JSON-in-CSV
format. Large histories preserve their exact order and contents via JSON
Pointers into corpus_audit.json; full events are not copied into each row.
"""
import json

INLINE_LIMIT = 8192
ENCODING = 'event_refs_v1'
AUDIT_FILE = 'corpus_audit.json'


class CompactReviewHistory:
    def __init__(self, records, audit_log, inline_limit=INLINE_LIMIT):
        self.inline_limit = inline_limit
        self.index = []
        self._refs = {}
        self._small = {}
        for i, event in enumerate(audit_log):
            self._refs.setdefault(id(event), f'/audit_log/{i}')
        for i, record in enumerate(records):
            for field in ('human_changes', 'quality_history'):
                for j, event in enumerate(record.get(field, [])):
                    self._refs.setdefault(id(event), f'/segments/{i}/{field}/{j}')

    def _small_event(self, event):
        key = id(event)
        if key not in self._small:
            # Check bounded chunks before building a complete JSON string.
            parts = []
            n = 0
            for part in json.JSONEncoder(ensure_ascii=False).iterencode(event):
                n += len(part)
                if n > self.inline_limit:
                    self._small[key] = None
                    break
                parts.append(part)
            else:
                self._small[key] = ''.join(parts)
        return self._small[key]

    def encode(self, events):
        parts = []
        size = 2  # brackets
        for event in events:
            part = self._small_event(event)
            if part is None or size + len(part) + (2 if parts else 0) > self.inline_limit:
                break
            size += len(part) + (2 if parts else 0)
            parts.append(part)
        else:
            return '[' + ', '.join(parts) + ']'
        # Each pointer targets an original event already in the audit. The
        # per-row index records order (including duplicate events) without
        # expanding the full history again in CSV, JSON and HTML.
        references = [self._refs[id(event)] for event in events]
        index = len(self.index)
        self.index.append(references)
        return json.dumps([{'encoding': ENCODING,
                            '$ref': f'{AUDIT_FILE}#/review_history/{index}',
                            'event_count': len(references)}], ensure_ascii=False)


def resolve_pointer(data, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Invalid audit JSON pointer.')
    for key in pointer[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        if isinstance(data, list):
            if not key.isdigit():
                raise ValueError('Invalid audit array index.')
            data = data[int(key)]
        else:
            data = data[key]
    return data


def expand_change_log(value, audit):
    """Recover exactly the old inline representation, leaving normal cells alone."""
    events = json.loads(value)
    if not (isinstance(events, list) and len(events) == 1 and isinstance(events[0], dict)
            and events[0].get('encoding') == ENCODING):
        return value
    marker = events[0]
    ref = marker['$ref']
    prefix = AUDIT_FILE + '#/review_history/'
    if not ref.startswith(prefix) or not ref[len(prefix):].isdigit():
        raise ValueError('Invalid compact history reference.')
    references = resolve_pointer(audit, ref.split('#', 1)[1])
    if not isinstance(references, list) or len(references) != marker['event_count']:
        raise ValueError('Incomplete compact history index.')
    if any(not (p.startswith('/audit_log/') or p.startswith('/segments/')) for p in references):
        raise ValueError('Invalid change event pointer.')
    return json.dumps([resolve_pointer(audit, p) for p in references], ensure_ascii=False)
