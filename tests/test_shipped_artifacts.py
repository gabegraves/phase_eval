import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


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
