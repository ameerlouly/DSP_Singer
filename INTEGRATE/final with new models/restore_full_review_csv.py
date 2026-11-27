"""Restore the full original 16-column review CSV from a V30 corpus export.

Usage: python restore_full_review_csv.py "path/to/export/folder"
       python restore_full_review_csv.py "path/to/export/folder" --gzip
Requires only Python's standard library. Never changes the original export.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from compact_review_history import expand_change_log


def restore(folder, compressed=False):
    folder = Path(folder)
    source = folder / 'review_compatible.csv'
    audit = json.loads((folder / 'corpus_audit.json').read_text(encoding='utf-8'))
    target = folder / ('review_full.csv.gz' if compressed else 'review_full.csv')
    # Accept legacy input cells of any practical size on Windows as well.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    opener = gzip.open if compressed else open
    created = False
    try:
        with source.open(encoding='utf-8-sig', newline='') as src:
            reader = csv.DictReader(src)
            if not reader.fieldnames or 'change_log' not in reader.fieldnames:
                raise ValueError('Input does not have a change_log column.')
            with opener(target, 'xt' if compressed else 'x', encoding='utf-8-sig', newline='') as dst:
                created = True
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row['change_log'] = expand_change_log(row['change_log'], audit)
                    writer.writerow(row)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', help='Folder containing the exported CSV and corpus_audit.json')
    parser.add_argument('--gzip', action='store_true', help='Write a compressed full CSV instead')
    args = parser.parse_args()
    print(restore(args.folder, args.gzip))
