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


@pytest.mark.parametrize('mode', ['once', 'log'])
def test_collections_refuse_records_mutated_after_validation(tmp_path, mode):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode
    from phaseeval.workbench.model import Unparsed

    rows = Ledger(tmp_path).collection('records', Unparsed, Mode(mode), KeyShape(('safe_id',)), owner=__name__)
    record = Unparsed(raw={}).model_copy(update={'verb': 'wrong'})
    with pytest.raises(ValidationError):
        if mode == 'once':
            rows.put(('one',), record)
        else:
            rows.append(('one',), record, at=datetime(2026, 1, 1, tzinfo=UTC))
    assert not (tmp_path / 'records').exists()
    valid = Unparsed(raw={})
    if mode == 'once':
        rows.put(('one',), valid)
        assert rows.get(('one',)) == valid
    else:
        rows.append(('one',), valid, at=datetime(2026, 1, 1, tzinfo=UTC))
        assert rows.entries(('one',)) == (valid,)


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


@pytest.mark.parametrize('kind', ['record', 'bytes', 'markers'])
def test_a_second_claim_fails_at_wiring(tmp_path, kind):
    from phaseeval.kernel.ledger import KeyShape, Mode

    ledger, _, Entry = collection(tmp_path)
    with pytest.raises(ValueError):
        if kind == 'markers':
            ledger.markers('entries', KeyShape(('safe_id',)), owner='other')
        else:
            ledger.collection('entries', bytes if kind == 'bytes' else Entry, Mode.ONCE,
                              KeyShape(('safe_id',)), owner='other')


@pytest.mark.parametrize('explicit_first', [False, True])
@pytest.mark.parametrize('mode', ['once', 'log', 'bytes', 'markers'])
def test_equivalent_storage_patterns_cannot_have_two_writers(tmp_path, explicit_first, mode):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode
    from phaseeval.kernel.records import Record

    ledger = Ledger(tmp_path)
    shape = KeyShape(('safe_id',))
    names = ('records', 'records/{0}.json' if mode == 'once' else 'records/{0}')
    if explicit_first:
        names = names[::-1]

    def claim(name, owner):
        if mode == 'markers' and '{' not in name:
            return ledger.markers(name, shape, owner=owner)
        return ledger.collection(name, bytes if mode in ('bytes', 'markers') else Record,
                                 Mode.LOG if mode == 'log' else Mode.ONCE, shape, owner=owner)

    claim(names[0], 'first')
    with pytest.raises(ValueError, match='already claimed'):
        claim(names[1], 'second')


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('name,mode,kind', [
    ('records/{0}', 'once', 'bytes'),
    ('records/{0}.json', 'once', 'record'),
    ('records/{0}/children', 'once', 'bytes'),
    ('records/{0}', 'log', 'record'),
    ('records/one.json', 'once', 'fixed'),
])
def test_intersecting_disk_namespaces_are_refused_at_wiring(tmp_path, reverse, name, mode, kind):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode
    from phaseeval.workbench.model import Unparsed

    ledger = Ledger(tmp_path)
    claims = [('records', Unparsed, Mode.ONCE, KeyShape(('safe_id',))),
              (name, Unparsed if kind == 'record' else bytes, Mode(mode),
               KeyShape(()) if kind == 'fixed' else KeyShape(('safe_id',)))]
    if reverse:
        claims.reverse()
    ledger.collection(*claims[0], owner='first')
    with pytest.raises(ValueError, match='already claimed'):
        ledger.collection(*claims[1], owner='second')


@pytest.mark.parametrize('mode', ['once', 'log'])
@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
@pytest.mark.parametrize('operation', ['read', 'write'])
def test_collections_refuse_nested_nonfinite_json_on_read_and_write(tmp_path, mode, value, operation):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode
    from phaseeval.workbench.model import Unparsed

    rows = Ledger(tmp_path).collection('records', Unparsed, Mode(mode), KeyShape(('safe_id',)), owner=__name__)
    if operation == 'write':
        record = Unparsed(raw={'values': [{'value': 1.0}]})
        record.raw['values'][0]['value'] = float(value)
        with pytest.raises(ValueError):
            if mode == 'once':
                rows.put(('one',), record)
            else:
                rows.append(('one',), record, at=datetime(2026, 1, 1, tzinfo=UTC))
        assert not (tmp_path / 'records').exists()
        return
    data = ('{"verb":"unparsed","raw":{"values":[{"value":' + value + '}]}}').encode()
    path = (tmp_path / 'records/one.json' if mode == 'once' else
            tmp_path / 'records/one' / f'20260101T000000000000Z-{hashlib.sha256(data).hexdigest()}.json')
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    with pytest.raises(ValueError):
        rows.get(('one',)) if mode == 'once' else rows.entries(('one',))


@pytest.mark.parametrize('kind', ['blobs', 'scratch'])
@pytest.mark.parametrize('explicit_first', [False, True])
def test_native_storage_patterns_cannot_be_claimed_by_a_collection(tmp_path, kind, explicit_first):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode

    ledger = Ledger(tmp_path)

    def native():
        getattr(ledger, kind)(owner='native')

    def explicit():
        ledger.collection(kind + '/{0}' + ('/owner.json' if kind == 'scratch' else ''),
                          bytes, Mode.ONCE, KeyShape(('safe_id',)), owner='collection')

    first, second = (explicit, native) if explicit_first else (native, explicit)
    first()
    with pytest.raises(ValueError, match='already claimed'):
        second()


def test_reordered_template_keys_are_refused_at_wiring(tmp_path):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode

    with pytest.raises(ValueError, match='placeholder'):
        Ledger(tmp_path).collection('rows/{1}/{0}.json', bytes, Mode.ONCE,
                                    KeyShape(('safe_id', 'safe_id')), owner=__name__)


def test_template_keys_roundtrip_with_prefixes(tmp_path):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode

    rows = Ledger(tmp_path).collection('rows/{0}/{1}.json', bytes, Mode.ONCE,
                                       KeyShape(('safe_id', 'safe_id')), owner=__name__)
    rows.put(('a', 'b'), b'value')
    assert tuple(rows.keys()) == (('a', 'b'),)
    assert tuple(rows.keys(('a',))) == (('a', 'b'),)
    assert rows.get(next(rows.keys())) == b'value'


def test_a_legacy_reader_does_not_take_the_writer_claim(tmp_path):
    from phaseeval.kernel.ledger import KeyShape, Ledger, Mode
    from phaseeval.kernel.records import Record

    ledger = Ledger(tmp_path)
    shape = KeyShape(('safe_id',))
    reader = ledger.legacy('records/{0}.json', bytes, reader='reader', key=shape)
    writer = ledger.collection('records', Record, Mode.ONCE, shape, owner='writer')
    writer.put(('one',), Record())
    assert reader.get(('one',)) == b'{}\n'
    assert writer.get(('one',)) == Record()
    with pytest.raises(ValueError, match='already claimed'):
        ledger.collection('records/{0}.json', Record, Mode.ONCE, shape, owner='other')


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


def test_scratch_owner_over_the_cap_is_never_published(tmp_path):
    from phaseeval.kernel.errors import TooLarge
    from phaseeval.kernel.ledger import Ledger, Limits

    scratch = Ledger(tmp_path, Limits(max_record_bytes=1)).scratch(owner=__name__)
    with pytest.raises(TooLarge):
        with scratch.workdir():
            pass
    assert not (tmp_path / 'scratch').exists()


@pytest.mark.parametrize('cleanup', ['workdir', 'sweep'])
def test_sweep_retries_interrupted_scratch_deletion_outside_the_lock(tmp_path, monkeypatch, cleanup):
    import shutil
    from phaseeval.kernel.ledger import Ledger, ScratchOwner
    from phaseeval.kernel.lock import ProcessOwner
    from phaseeval.kernel.records import canonical_bytes

    ledger = Ledger(tmp_path)
    scratch = ledger.scratch(owner=__name__)
    remove = shutil.rmtree

    def interrupted(path):
        with ledger.transaction():
            pass
        raise OSError('interrupted deletion')

    with monkeypatch.context() as patch:
        patch.setattr(shutil, 'rmtree', interrupted)
        with pytest.raises(OSError, match='interrupted deletion'):
            if cleanup == 'workdir':
                with scratch.workdir() as workdir:
                    (workdir / 'partial').write_bytes(b'partial result')
            else:
                workdir = tmp_path / 'scratch' / 'dead'
                workdir.mkdir(parents=True)
                owner = ProcessOwner.current().model_copy(update={'boot_id': 'previous-boot'})
                (workdir / 'owner.json').write_bytes(canonical_bytes(ScratchOwner(owner=owner)))
                scratch.sweep()
    assert not workdir.exists()
    assert len(tuple((tmp_path / 'scratch').glob('.trash-*'))) == 1

    def unlocked_remove(path):
        with ledger.transaction():
            pass
        remove(path)

    monkeypatch.setattr(shutil, 'rmtree', unlocked_remove)
    recovered = Ledger(tmp_path).scratch(owner=__name__)
    assert recovered.sweep() == 1
    assert tuple((tmp_path / 'scratch').iterdir()) == ()
    assert recovered.sweep() == 0


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
