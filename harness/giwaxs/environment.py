"""Owned GIWAXS evidence with bounded native reads and immutable reduction lineage."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from harness.agent.evidence import checked_bytes
from harness.agent.episode import atomic_json, digest, strict_json
from phaseeval.giwaxs import ACTION_TYPES, ArtifactReceipt, GiwaxsManifest, Runtime, SubmissionReceipt, EvidenceRegistry

EXPECTED_RUNTIME = {'python': '3.10.15', 'pyFAI': '2023.3.0', 'pygix': '2022.11.14',
                    'numpy': '1.26.4', 'fabio': '2025.10.0', 'scipy': '1.11.4', 'silx': '1.1.2'}
SOURCE_FILES = ('harness/giwaxs/environment.py', 'harness/giwaxs/worker.py',
                'harness/giwaxs/requirements.txt', 'src/phaseeval/giwaxs.py')
WORKER = Path(__file__).with_name('worker.py').resolve()


def validate_runtime(value):
    value = Runtime.model_validate(value).model_dump()
    if value != EXPECTED_RUNTIME:
        raise ValueError(f'GIWAXS runtime differs from pinned reproduction: {value}')
    return value


def probe_runtime(python, execute):
    return validate_runtime(strict_json(execute([str(python), str(WORKER), '--probe'])))


def arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if any(a.dtype.kind not in 'biuf' or not np.isfinite(a).all() for a in result.values()):
        raise ValueError('numeric evidence must be finite numeric arrays')
    return result


def validate_map(value):
    if set(value) != {'intensity', 'count', 'qr', 'qz'}:
        raise ValueError('map arrays differ from contract')
    if any(a.dtype.kind not in 'biuf' or not np.isfinite(a).all() for a in value.values()):
        raise ValueError('map arrays must be finite numeric values')
    if any(value[k].ndim != 1 for k in ('qr', 'qz')):
        raise ValueError('map coordinates must be one-dimensional')
    shape = (len(value['qz']), len(value['qr']))
    if min(shape) < 2 or any(np.any(np.diff(value[k]) <= 0) for k in ('qr', 'qz')):
        raise ValueError('map coordinates must strictly increase')
    if value['intensity'].shape != shape or value['count'].shape != shape or np.any(value['count'] < 0):
        raise ValueError('map shape or coverage is invalid')


def support_summary(x, usable):
    indexes = np.flatnonzero(usable)
    return {'bounding_extent': [float(x[indexes[0]]), float(x[indexes[-1]])] if len(indexes) else None,
            'contiguous_runs': int(1 + np.count_nonzero(np.diff(indexes) != 1)) if len(indexes) else 0,
            'extent_meaning': 'Bounding coordinates only; internal gaps may be unusable.'}


def numeric_summary(value, kind, coordinate=None, *, mask=False):
    intensity = value['intensity']
    summary = {'scope': 'full_artifact', 'shape': list(intensity.shape), 'total_bins': int(intensity.size)}
    if kind == 'detector':
        summary['value_extent'] = [float(intensity.min()), float(intensity.max())]
        if mask:
            summary.update(masked_pixels=int(np.count_nonzero(intensity)),
                           unmasked_pixels=int(np.count_nonzero(intensity == 0)),
                           mask_convention='Nonzero pixels are masked when mask_policy=supplied.')
        return summary
    usable = value['count'] > 0 if kind == 'map' else value['usable'].astype(bool)
    values = intensity[usable]
    summary.update(usable_bins=int(usable.sum()), unusable_bins=int((~usable).sum()),
                   usable_intensity_extent=[float(values.min()), float(values.max())] if values.size else None)
    if kind == 'map':
        summary.update(coordinate_definition='q=2pi/d', unit='nm^-1',
                       coordinate_extents={k: [float(value[k][0]), float(value[k][-1])] for k in ('qr', 'qz')},
                       covered_zero_bins=int(np.count_nonzero(values == 0)),
                       covered_negative_bins=int(np.count_nonzero(values < 0)),
                       coverage_meaning='count > 0 is covered; measured zero/negative intensity is retained. Extents do not imply coverage.')
    else:
        summary.update(coordinate=coordinate, output_axis_extent=[float(value['x'][0]), float(value['x'][-1])],
                       usable_support=support_summary(value['x'], usable),
                       covered_count_extent=[int(value['covered_count'].min()), int(value['covered_count'].max())])
    return summary


def comparison_summary(value):
    indexes = value['common_left_indexes']
    runs = 1 + int(np.count_nonzero(np.diff(indexes) != 1)) if indexes else 0
    return {k: v for k, v in value.items() if k != 'common_left_indexes'} | {
        'contiguous_runs': runs, 'common_interval_is_contiguous': runs == 1,
        'interval_meaning': 'Bounding coordinates of common supported left bins; may contain gaps.'}


def selected_indexes(value, kind, selection):
    """Select native bins inclusively; offsets page the selection without dropping rows."""
    def within(coordinates, lower, upper):
        if lower > upper or lower < coordinates[0] or upper > coordinates[-1]:
            raise ValueError('selection bounds reversed or outside native coordinates')
        return (coordinates >= lower) & (coordinates <= upper)

    if kind == 'profile':
        indexes = np.flatnonzero(within(value['x'], selection.lower, selection.upper))
    else:
        if kind == 'map':
            rows = within(value['qz'], selection.qz_min, selection.qz_max)
            columns = within(value['qr'], selection.qr_min, selection.qr_max)
        else:
            rows = within(np.arange(value['intensity'].shape[0]), selection.row_min, selection.row_max)
            columns = within(np.arange(value['intensity'].shape[1]), selection.column_min, selection.column_max)
        indexes = (np.flatnonzero(rows)[:, None] * value['intensity'].shape[1] + np.flatnonzero(columns)).ravel()
    if not len(indexes):
        raise ValueError('selection contains no native bins')
    return indexes


def extract_profile(value, *, axis, lower, upper, mode, coordinate_definition):
    validate_map(value)
    strip = value['qz' if axis == 'qr' else 'qr']
    if lower < strip[0] or upper > strip[-1] or lower >= upper:
        raise ValueError('ROI is empty, reversed or outside map')
    a, b = (int(np.argmin(abs(strip - q))) for q in (lower, upper))
    if a >= b:
        raise ValueError('ROI collapses to an empty nearest-bin slice')
    intensity = value['intensity'] if axis == 'qr' else value['intensity'].T
    count = value['count'] if axis == 'qr' else value['count'].T
    selected, covered = intensity[a:b], count[a:b] > 0
    n = covered.sum(axis=0)
    if mode == 'notebook':
        y = selected.mean(axis=0)
        usable = n == b-a
    else:
        y = np.divide(np.where(covered, selected, 0).sum(axis=0), n,
                      out=np.zeros(n.shape, dtype=float), where=n > 0)
        usable = n > 0
    x = value[axis].copy()
    if coordinate_definition == 's=1/d':
        x = x / (2*np.pi)
    return {'x': x, 'intensity': y, 'covered_count': n, 'usable': usable}, {
        'roi_axis': 'qz' if axis == 'qr' else 'qr', 'output_axis_extent': [float(x[0]), float(x[-1])],
        'usable_support': support_summary(x, usable),
        'requested_roi': [lower, upper], 'roi_coordinate_definition': 'q=2pi/d', 'roi_unit': 'nm^-1',
        'slice': [a, b], 'selected_rows': b-a, 'realized_roi': [float(strip[a]), float(strip[b-1])],
        'excluded_upper_coordinate': float(strip[b]), 'selection': 'nearest-bin_upper-exclusive',
        'aggregation': 'unweighted_mean', 'mode': mode, 'uncovered_bins': int((n == 0).sum()),
        'partial_coverage_bins': int(((n > 0) & (n < b-a)).sum())}


def compare_profiles(left, right, lower=None, upper=None):
    """Interpolate only within contiguous supported runs of the right profile."""
    x, lx, rx = left['x'], left['intensity'], right['x']
    valid = left['usable'].astype(bool).copy()
    if lower is not None:
        valid &= x >= lower
    if upper is not None:
        valid &= x <= upper
    interpolated = np.zeros(len(x))
    support = np.zeros(len(x), dtype=bool)
    indexes = np.flatnonzero(right['usable'])
    for run in np.split(indexes, np.flatnonzero(np.diff(indexes) != 1) + 1):
        if not len(run):
            continue
        eligible = (x >= rx[run[0]]) & (x <= rx[run[-1]])
        interpolated[eligible] = np.interp(x[eligible], rx[run], right['intensity'][run])
        support |= eligible
    valid &= support
    if not valid.any():
        raise ValueError('profiles have no common covered support')
    delta = interpolated[valid] - lx[valid]
    denominator = float(np.linalg.norm(lx[valid]))
    return {'common_bins': int(valid.sum()), 'left_total_bins': len(x),
            'excluded_left_bins': int((~valid).sum()), 'right_usable_bins': int(right['usable'].sum()),
            'common_left_indexes': np.flatnonzero(valid).tolist(),
            'common_interval': [float(x[valid][0]), float(x[valid][-1])],
            'interpolation': 'linear_on_left_grid_within_contiguous_right_support_only',
            'relative_l2_change': float(np.linalg.norm(delta)/denominator) if denominator else None,
            'max_absolute_change': float(np.max(abs(delta))), 'scientific_reward': None}


class GiwaxsEvidence:
    def __init__(self, inputs: Path, case_id: str, output: Path, trial: Path, *, python=None, execute=None, runtime=None):
        self.inputs, self.output, self.trial = Path(inputs).absolute(), Path(output), Path(trial)
        manifest = checked_bytes(self.inputs, 'manifest.json')
        parsed = GiwaxsManifest.model_validate(strict_json(manifest))
        cases = [c for c in parsed.cases if c.case_id == case_id]
        if len(cases) != 1:
            raise ValueError('GIWAXS manifest must have exactly one matching case')
        self.case = cases[0].model_dump()
        self.manifest_sha256 = digest(manifest)
        self.bound_inputs = {'manifest.json': self.manifest_sha256}
        self.files, self.artifacts, self.candidates = {}, {}, []
        self.python, self.execute, self.runtime = python, execute, validate_runtime(runtime) if runtime is not None else None
        self.submission, self.reduction_bound = None, False
        self.labels = set()
        self.native = {}
        for name, suffixes in (('raw', {'.tif', '.tiff'}), ('geometry', {'.poni'}), ('mask', {'.edf'})):
            ref = self.case[name]
            if Path(ref['file']).suffix.lower() not in suffixes:
                raise ValueError(f'{name} file has unsupported type')
            raw = checked_bytes(self.inputs, ref['file'], ref['sha256'])
            if not raw:
                raise ValueError('input evidence is empty')
            self.bound_inputs[ref['file']] = ref['sha256']
            self.files[name] = (self.inputs, ref['file'], 'text' if name == 'geometry' else 'detector', ref['sha256'])

    def register_output(self, extra):
        name, sha = extra['executor_output'], extra['executor_sha256']
        checked_bytes(self.trial, name, sha)
        self.files[extra['evidence_handle']] = (self.trial, name, 'text', sha)
        self._save_registry()

    def _check_inputs(self):
        for name, sha in self.bound_inputs.items():
            checked_bytes(self.inputs, name, sha)

    def _register_context(self):
        self._check_inputs()
        if 'context' in self.files:
            return
        if any(p.is_symlink() for p in (self.trial, *self.trial.parents)):
            raise ValueError('symlink trial directory')
        self.trial.mkdir(parents=True, exist_ok=True)
        name = 'giwaxs-context.json'
        path = self.trial / name
        expected = (json.dumps(self.case, indent=2, allow_nan=False) + '\n').encode()
        if not path.exists() and not path.is_symlink():
            atomic_json(path, self.case)
        checked_bytes(self.trial, name, digest(expected))
        self.files['context'] = (self.trial, name, 'text', digest(expected))
        try:
            self._save_registry()
        except Exception:
            del self.files['context']
            raise

    def _bytes(self, handle, seen=None):
        if handle not in self.files:
            raise ValueError(f'unknown evidence handle: {repr(handle)[:160]}')
        self._check_inputs()
        if handle == 'context':
            expected = (json.dumps(self.case, indent=2, allow_nan=False) + '\n').encode()
            if self.files[handle][3] != digest(expected):
                raise ValueError('context digest differs from manifest case')
        seen = set() if seen is None else seen
        if handle in seen:
            raise ValueError('cyclic artifact lineage')
        seen.add(handle)
        if handle.endswith(':receipt') and handle[:-8] in self.artifacts:
            self._bytes(handle[:-8], seen.copy())
        if handle in self.artifacts:
            for parent, sha in self.artifacts[handle]['parents'].items():
                if digest(self._bytes(parent, seen.copy())) != sha:
                    raise ValueError('parent digest changed')
            receipt = self.files[handle + ':receipt']
            checked_bytes(receipt[0], receipt[1], receipt[3])
        root, name, _kind, sha = self.files[handle]
        return checked_bytes(root, name, sha)

    def _save_registry(self):
        if any(p.is_symlink() for p in (self.trial, *self.trial.parents)):
            raise ValueError('symlink trial directory')
        self.trial.mkdir(parents=True, exist_ok=True)
        atomic_json(self.trial / 'giwaxs-evidence.json', EvidenceRegistry.model_validate({
            'schema_version': 1, 'task_kind': 'giwaxs', 'case_id': self.case['case_id'],
            'manifest_sha256': self.manifest_sha256, 'inputs_root': str(self.inputs), 'inputs': self.bound_inputs,
            'files': {handle: {'root': str(root.absolute()), 'file': name, 'kind': kind, 'sha256': sha}
                      for handle, (root, name, kind, sha) in self.files.items()},
            'artifacts': self.artifacts, 'scientific_reward': None, 'scientific_qualified': False}).model_dump())

    def _allocate(self, label):
        if label in self.labels:
            raise ValueError('action label already used')
        self.labels.add(label)
        if any(p.is_symlink() for p in (self.output, *self.output.parents)):
            raise ValueError('symlink output directory')
        self.output.mkdir(parents=True, exist_ok=True)
        directory = self.output / label
        directory.mkdir(exist_ok=False)
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise ValueError('symlink output directory')
        return directory

    def _worker(self, directory, request):
        if self.execute is None or self.python is None or self.runtime is None:
            raise RuntimeError('GIWAXS worker executor and validated runtime are required')
        self._check_inputs()
        target = directory / 'data.npz'
        request = request | {'inputs': str(self.inputs), 'output': str(target.absolute())}
        atomic_json(directory / 'request.json', request)
        try:
            reply = strict_json(self.execute([str(self.python), str(WORKER), str((directory / 'request.json').absolute())]))
            if validate_runtime(reply['runtime']) != self.runtime:
                raise ValueError('worker runtime changed')
            self._check_inputs()
            raw = checked_bytes(directory, 'data.npz', reply['sha256'])
            return arrays(raw), reply
        except Exception as error:
            raise RuntimeError(f'GIWAXS worker failed: {error}') from error

    def _register(self, directory, kind, parameters, coordinate, parents, warnings, payload=None):
        self._check_inputs()
        for parent, sha in parents.items():
            if digest(self._bytes(parent)) != sha:
                raise ValueError('parent digest changed')
        name = 'data.npz' if payload is None else 'comparison.json'
        if payload is not None:
            atomic_json(directory / name, payload)
        raw = checked_bytes(directory, name)
        handle = f'{kind}:{directory.name}'
        receipt = ArtifactReceipt(case_id=self.case['case_id'], kind=kind, handle=handle, parents=parents,
                                  inputs=self.bound_inputs, outputs={name: digest(raw)}, parameters=parameters,
                                  coordinate=coordinate, runtime=self.runtime, warnings=warnings).model_dump()
        atomic_json(directory / 'receipt.json', receipt)
        self.files[handle] = (directory, name, 'text' if payload is not None else kind, digest(raw))
        self.files[handle + ':receipt'] = (directory, 'receipt.json', 'text', digest(checked_bytes(directory, 'receipt.json')))
        self.artifacts[handle] = receipt
        try:
            self._save_registry()
        except Exception:
            del self.artifacts[handle], self.files[handle], self.files[handle + ':receipt']
            raise
        summary = comparison_summary(payload) if payload is not None else numeric_summary(arrays(raw), kind, coordinate)
        return {'handle': handle, 'receipt_handle': handle + ':receipt', 'summary': summary, **receipt}

    def read(self, **kwargs):
        args = ACTION_TYPES['read_evidence'].model_validate(kwargs)
        handle, start = args.handle, args.start
        self._check_inputs()
        if handle is None:
            raw = json.dumps({'handles': list(dict.fromkeys(['context', *self.files])), 'row_limit': 100, 'text_limit': 2000}).encode()
            handle, kind = 'catalog', 'text'
        elif handle == 'context':
            self._register_context()
            raw, kind = self._bytes(handle), 'text'
        else:
            raw = self._bytes(handle)
            kind = self.files[handle][2]
        if args.selection is not None and args.selection.kind != kind:
            raise ValueError('selection kind must match numeric evidence kind')
        result = {'handle': handle, 'kind': kind, 'sha256': digest(raw)}
        maximum = 2000 if kind == 'text' else 100
        limit = args.limit if args.limit is not None else maximum
        if limit > maximum:
            raise ValueError(f'limit exceeds {maximum}')
        if kind == 'text':
            if handle in self.artifacts and self.artifacts[handle]['kind'] == 'comparison':
                result['summary'] = comparison_summary(strict_json(raw))
            text = raw.decode('utf-8')
            if start > len(text):
                raise ValueError('start exceeds text length')
            end = min(start+limit, len(text))
            return result | {'text': text[start:end], 'total_chars': len(text), 'offset_unit': 'unicode_character',
                             'start': start, 'end': end, 'next_start': end if end < len(text) else None}
        if kind == 'detector':
            if handle not in self.native:
                directory = self._allocate(f'native.{handle}.{len(self.labels)}')
                value, reply = self._worker(directory, {'operation': 'decode', 'file': self.case[handle]})
                if set(value) != {'intensity'} or value['intensity'].ndim != 2:
                    raise RuntimeError('worker returned invalid detector array')
                self.native[handle] = (directory, reply['sha256'])
            directory, sha = self.native[handle]
            value = arrays(checked_bytes(directory, 'data.npz', sha))
            result['decoded_sha256'] = sha
            result['runtime'] = self.runtime
        else:
            value = arrays(raw)
        shape = value['intensity'].shape
        native_total = int(value['intensity'].size)
        selected = selected_indexes(value, kind, args.selection) if args.selection is not None else None
        total = len(selected) if selected is not None else native_total
        result['summary'] = numeric_summary(value, kind, self.artifacts.get(handle, {}).get('coordinate'), mask=handle == 'mask')
        if start > total:
            raise ValueError('start exceeds native row count')
        end = min(start+limit, total)
        indexes = selected[start:end].tolist() if selected is not None else list(range(start, end))
        if kind == 'detector':
            columns = ['row', 'column', 'value']
            rows = [[i//shape[1], i%shape[1], value['intensity'].flat[i].item()] for i in indexes]
            extra = {'coordinate_definition': 'detector_pixel', 'units': ['pixel', 'pixel', 'mask_flag' if handle == 'mask' else 'recorded_intensity']}
        elif kind == 'map':
            columns = ['qz', 'qr', 'intensity', 'coverage_count', 'usable']
            rows = [[float(value['qz'][i//shape[1]]), float(value['qr'][i%shape[1]]),
                     float(value['intensity'].flat[i]), float(value['count'].flat[i]), bool(value['count'].flat[i] > 0)] for i in indexes]
            extra = {'coordinate_definition': 'q=2pi/d', 'units': ['nm^-1', 'nm^-1', 'recorded_intensity', 'contribution_count', 'boolean']}
        else:
            columns = ['coordinate', 'intensity', 'covered_count', 'usable']
            rows = [[float(value['x'][i]), float(value['intensity'][i]), int(value['covered_count'][i]), bool(value['usable'][i])] for i in indexes]
            extra = {'coordinate': self.artifacts[handle]['coordinate'], 'units': ['nm^-1', 'recorded_intensity', 'covered_strip_rows', 'boolean']}
        return result | extra | {'shape': list(shape), 'columns': columns, 'rows': rows, 'row_indexes': indexes,
                                 'total_rows': total, 'native_total_rows': native_total,
                                 'selection': args.selection.model_dump() if args.selection is not None else None,
                                 'start': start, 'end': end, 'next_start': end if end < total else None}

    def call(self, name, arguments):
        if name not in ACTION_TYPES:
            raise ValueError('unknown GIWAXS action')
        args = ACTION_TYPES[name].model_validate(arguments).model_dump()
        if name == 'read_evidence':
            return self.read(**args)
        self._check_inputs()
        if self.submission is not None:
            raise ValueError('submission already completed')
        if name == 'submit':
            citations = {}
            for claim in args['claims']:
                for handle in claim['evidence']:
                    if handle == 'context':
                        self._register_context()
                    citations[handle] = digest(self._bytes(handle))
            if args['status'] == 'supported' and not any(h in self.artifacts for h in citations):
                raise ValueError('supported submission requires a derived artifact citation')
            self._save_registry()
            result = SubmissionReceipt.model_validate(args | {'case_id': self.case['case_id'], 'citations': citations}).model_dump()
            if any(p.is_symlink() for p in (self.output, *self.output.parents)):
                raise ValueError('symlink output directory')
            self.output.mkdir(parents=True, exist_ok=True)
            atomic_json(self.output / 'analysis.json', result)
            self.submission = result
            return result
        directory = self._allocate(args['label'])
        if name == 'transform':
            value, reply = self._worker(directory, {'operation': 'transform', 'case': self.case, 'parameters': args})
            try:
                validate_map(value)
            except (ValueError, TypeError, KeyError) as error:
                raise RuntimeError(f'GIWAXS worker returned invalid map: {error}') from error
            warnings = reply['warnings']
            if any(args[k] != self.case['acquisition'][k] for k in self.case['acquisition']) or args['distance_scale'] != 1:
                warnings.append('Alternative geometry is an exploratory assumption, not corrected acquisition truth.')
            result = self._register(directory, 'map', args | {'executed': reply},
                                    {'definition': 'q=2pi/d', 'unit': 'nm^-1', 'axis': 'qr'},
                                    {h: self.files[h][3] for h in ('raw', 'geometry', 'mask')}, warnings)
            self.reduction_bound = True
            return result
        if name == 'extract':
            parent = args['map_handle']
            if parent not in self.artifacts or self.artifacts[parent]['kind'] != 'map':
                raise ValueError('extract requires an owned map')
            raw = self._bytes(parent)
            value, details = extract_profile(arrays(raw), **{k: args[k] for k in ('axis', 'lower', 'upper', 'mode', 'coordinate_definition')})
            np.savez_compressed(directory / 'data.npz', **value)
            warnings = ['Notebook aggregation retains uncovered fill contributions; only fully covered profile bins are comparison-eligible.'] if args['mode'] == 'notebook' and (details['partial_coverage_bins'] or details['uncovered_bins']) else []
            return self._register(directory, 'profile', args | details | {'processing_numpy': np.__version__},
                                  {'definition': args['coordinate_definition'], 'unit': 'nm^-1', 'axis': args['axis']},
                                  {parent: digest(raw)}, warnings)
        left, right = args['left'], args['right']
        if any(h not in self.artifacts or self.artifacts[h]['kind'] != 'profile' for h in (left, right)):
            raise ValueError('compare requires owned profiles')
        if self.artifacts[left]['coordinate'] != self.artifacts[right]['coordinate']:
            raise ValueError('incompatible profile coordinate conventions')
        raw_left, raw_right = self._bytes(left), self._bytes(right)
        result = compare_profiles(arrays(raw_left), arrays(raw_right), args['lower'], args['upper'])
        return self._register(directory, 'comparison', args | {'processing_numpy': np.__version__}, self.artifacts[left]['coordinate'],
                              {left: digest(raw_left), right: digest(raw_right)}, [], result)
