import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_bundled_episode_ids_and_referenced_blob_sizes_are_valid():
    from phaseeval.workbench.exports import _record_id

    with tarfile.open(REPO / 'launcher/demo/phaseeval-demo.tar.xz') as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    rows = [json.loads(line) for line in files['dataset/episodes.jsonl'].splitlines()]
    assert len(rows) == 6
    assert all(row['record_id'] == _record_id({k: v for k, v in row.items() if k != 'record_id'})
               for row in rows), 'bundled episode record_id disagrees with its content'

    def check(value):
        if isinstance(value, dict):
            if 'sha256' in value and 'size_bytes' in value:
                assert value['size_bytes'] == len(files['store/blobs/' + value['sha256']])
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    for name, data in files.items():
        if name.endswith(('.json', '.jsonl')) and not name.startswith('store/blobs/'):
            for doc in data.splitlines() if name.endswith('.jsonl') else [data]:
                if doc.strip():
                    check(json.loads(doc))


def test_no_member_of_the_demo_archive_matches_a_leak_rule():
    from phaseeval.publish.leaks import PUBLIC_RULES, scan_bytes

    # Demo identities are intentional fixture data; submissions must still refuse them.
    rules = [rule for rule in PUBLIC_RULES if rule.name != 'demo_run']
    with tarfile.open(REPO / 'launcher/demo/phaseeval-demo.tar.xz') as archive:
        members = [m for m in archive if m.isfile()]
        assert len(members) == 1761
        hits = {m.name: scan_bytes(archive.extractfile(m).read(), rules=rules) for m in members}
    assert not {name: findings for name, findings in hits.items() if findings}


def test_no_shipped_data_file_matches_a_leak_rule():
    from phaseeval.publish.leaks import PUBLIC_RULES, scan_bytes

    rules = [rule for rule in PUBLIC_RULES if rule.name != 'demo_run']
    hits = {str(path.relative_to(REPO)): scan_bytes(path.read_bytes(), rules=rules)
            for path in (REPO / 'data').rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    assert hits
    assert not {name: findings for name, findings in hits.items() if findings}
