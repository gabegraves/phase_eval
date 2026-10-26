import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError


def collection(tmp_path, *, mode=None, maximum=1024):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Limits, Mode
    from phaseeval.kernel.records import Record

    class Entry(Record):
        text: str
        created_at_utc: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    ledger = Ledger(tmp_path, Limits(max_record_bytes=maximum))
    return ledger, ledger.collection('entries', Entry, mode or Mode.ONCE, KeyShape(('safe_id',)), owner=__name__), Entry


def test_publish_once_refuses_different_bytes(tmp_path):
    from phaseeval.kernel.errors import Conflict

    _, rows, Entry = collection(tmp_path)
    assert not rows.put(('one',), Entry(text='first')).existed
    assert rows.put(('one',), Entry(text='first')).existed
    with pytest.raises(Conflict):
        rows.put(('one',), Entry(text='second'))
    assert rows.get(('one',)).text == 'first'


def test_append_deduplicates_and_checks_the_digest(tmp_path):
    from phaseeval.kernel.ledger import Mode

    _, rows, Entry = collection(tmp_path, mode=Mode.LOG)
    first = rows.append(('one',), Entry(text='first'))
    assert rows.append(('one',), Entry(text='first')).existed
    assert rows.entries(('one',)) == (Entry(text='first'),)
    (tmp_path / first.path).write_bytes(b'{}\n')
    with pytest.raises(ValueError):
        rows.entries(('one',))


def test_log_names_use_utc_and_refuse_naive_timestamps(tmp_path):
    from phaseeval.kernel.ledger import Mode

    _, rows, Entry = collection(tmp_path, mode=Mode.LOG)
    entry = Entry(text='offset', created_at_utc=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=5))))
    assert rows.append(('one',), entry).path.name.startswith('20251231T190000000000Z-')
    with pytest.raises(ValueError, match='timezone'):
        rows.append(('naive',), Entry(text='naive', created_at_utc=datetime(2026, 1, 1)))
    assert rows.entries(('naive',)) == ()


@pytest.mark.parametrize('key', ['..', '/escape', 'a/b', 'a\\b', '', 'a\x00b'])
def test_invalid_segments_never_touch_disk(tmp_path, key):
    _, rows, Entry = collection(tmp_path)
    with pytest.raises(ValueError):
        rows.put((key,), Entry(text='x'))
    assert list(tmp_path.iterdir()) == []


def test_a_record_over_the_cap_is_never_written(tmp_path):
    from phaseeval.kernel.errors import TooLarge

    _, rows, Entry = collection(tmp_path, maximum=128)
    with pytest.raises(TooLarge):
        rows.put(('big',), Entry(text='x' * 129))
    assert rows.get(('big',)) is None
    rows.put(('small',), Entry(text='ok'))
    (tmp_path / 'entries/small.json').write_bytes(b'x' * 129)
    with pytest.raises(TooLarge):
        rows.get(('small',))


def test_log_and_blob_caps_apply_before_writing_and_on_read(tmp_path):
    from phaseeval.kernel.errors import TooLarge
    from phaseeval.kernel.ledger import Ledger, Limits, Mode

    _, rows, Entry = collection(tmp_path / 'records', mode=Mode.LOG, maximum=128)
    with pytest.raises(TooLarge):
        rows.append(('big',), Entry(text='x' * 129))
    assert rows.entries(('big',)) == ()
    published = rows.append(('small',), Entry(text='ok'))
    (tmp_path / 'records' / published.path).write_bytes(b'x' * 129)
    with pytest.raises(TooLarge):
        rows.entries(('small',))
    blobs = Ledger(tmp_path / 'blobs', Limits(max_blob_bytes=4)).blobs(owner=__name__)
    with pytest.raises(TooLarge):
        blobs.put(b'12345')
    ref = blobs.put(b'1234')
    (tmp_path / 'blobs/blobs' / ref).write_bytes(b'12345')
    with pytest.raises(TooLarge):
        blobs.get(ref)


def test_a_second_claim_fails_at_wiring(tmp_path):
    from phaseeval.kernel.ledger import KeyShape, Mode

    ledger, _, Entry = collection(tmp_path)
    with pytest.raises(ValueError):
        ledger.collection('entries', Entry, Mode.ONCE, KeyShape(('safe_id',)), owner='other')


def test_markers_validate_empty_files_on_membership_reads(tmp_path):
    from phaseeval.kernel.errors import TooLarge
    from phaseeval.kernel.ledger import KeyShape, Ledger

    markers = Ledger(tmp_path).markers('markers', KeyShape(('safe_id',)), owner=__name__)
    assert not markers.contains(('one',))
    published = markers.add(('one',))
    assert markers.contains(('one',))
    assert markers.members(()) == ('one',)
    (tmp_path / published.path).write_bytes(b'not a marker')
    with pytest.raises(TooLarge):
        markers.contains(('one',))
    with pytest.raises(TooLarge):
        markers.members(())


def test_separate_ledgers_share_the_nonreentrant_transaction(tmp_path):
    from phaseeval.kernel.ledger import Ledger

    one, two = Ledger(tmp_path), Ledger(tmp_path / 'child/..')
    with one.transaction():
        two.assert_locked()
        with pytest.raises(RuntimeError):
            with two.transaction():
                pass


def test_blobs_refuse_corruption_and_symlinks_at_any_depth(tmp_path):
    from phaseeval.kernel.ledger import Ledger

    blobs = Ledger(tmp_path / 'store').blobs(owner=__name__)
    ref = blobs.put(b'content')
    assert ref == hashlib.sha256(b'content').hexdigest()
    assert blobs.get(ref) == b'content'
    (tmp_path / 'store/blobs' / ref).write_bytes(b'changed')
    with pytest.raises(ValueError):
        blobs.get(ref)
    (tmp_path / 'store/blobs').rename(tmp_path / 'outside')
    (tmp_path / 'store/blobs').symlink_to(tmp_path / 'outside', target_is_directory=True)
    with pytest.raises(ValueError):
        blobs.put(b'another')


def test_legacy_reads_are_capped_and_never_writable(tmp_path):
    from phaseeval.kernel.errors import TooLarge
    from phaseeval.kernel.ledger import KeyShape, Ledger, Limits

    path = tmp_path / 'old'
    path.mkdir()
    (path / 'record').write_bytes(b'12345')
    rows = Ledger(tmp_path, Limits(max_record_bytes=4)).legacy('old/{0}', bytes, reader=__name__, key=KeyShape(('safe_id',)))
    with pytest.raises(TooLarge):
        rows.get(('record',))
    with pytest.raises(TypeError):
        rows.put(('new',), b'ok')
    assert not (path / 'new').exists()


def test_scratch_is_owner_stamped_and_removed_on_exception(tmp_path):
    from phaseeval.kernel.ledger import Ledger, ScratchOwner

    scratch = Ledger(tmp_path).scratch(owner=__name__)
    with pytest.raises(RuntimeError, match='executor failed'):
        with scratch.workdir() as workdir:
            assert ScratchOwner.model_validate_json((workdir / 'owner.json').read_bytes()).owner.alive()
            (workdir / 'result').write_bytes(b'partial')
            raise RuntimeError('executor failed')
    assert not workdir.exists()


def test_scratch_sweep_preserves_live_owners_and_recorded_children(tmp_path):
    from phaseeval.kernel.ledger import Ledger, ScratchOwner
    from phaseeval.kernel.lock import ProcessOwner
    from phaseeval.kernel.records import canonical_bytes

    scratch = Ledger(tmp_path).scratch(owner=__name__)
    owner = ProcessOwner.current()
    dead = owner.model_copy(update={'boot_id': 'previous-boot'})
    for name, identity in [('live', owner), ('dead', dead), ('child', dead)]:
        workdir = tmp_path / 'scratch' / name
        workdir.mkdir(parents=True)
        (workdir / 'owner.json').write_bytes(canonical_bytes(ScratchOwner(owner=identity, slot='a'*64 if name == 'child' else None)))
    scratch.child_alive = lambda slot: slot == 'a'*64
    assert scratch.sweep() == 1
    assert sorted(p.name for p in (tmp_path / 'scratch').iterdir()) == ['child', 'live']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_actual_records_refuse_nonfinite_numbers(value):
    from phaseeval.workbench.model import Cell

    with pytest.raises(ValidationError):
        Cell(a=value, b=None, c=None, alpha=None, beta=None, gamma=None)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_gsasii_reader_records_refuse_nonfinite_json(value):
    from phaseeval.workbench.importers.gsasii import GpxAnalysis

    payload = ('{"source_sha256":"' + 'a'*64 + '","saved_using":null,"reader_gsasii_version":null,'
               '"rwp_percent":' + value + ',"gof":null,"converged":null,"phases":[],"histograms":[]}')
    with pytest.raises(ValidationError):
        GpxAnalysis.model_validate_json(payload)
