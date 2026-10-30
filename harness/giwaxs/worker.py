"""Standalone Python 3.10 worker. Only detector decoding and reciprocal transform."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import warnings


def runtime():
    return {'python': platform.python_version(), **{name: version(name) for name in ('pyFAI', 'pygix', 'numpy', 'fabio', 'scipy', 'silx')}}


def checked(root, ref):
    relative = Path(ref['file'])
    if relative.is_absolute() or any(p.startswith('.') for p in relative.parts):
        raise ValueError('unsafe worker input')
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('symlink or escaping worker input')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('worker input digest changed')
    return path


def validate_geometry(path):
    # ponytail: only the deposited self-contained PONI v2 Detector form is supported;
    # auxiliary files need manifest entries and the same path/hash checks before use.
    fields = {}
    allowed = {'poni_version', 'detector', 'detector_config', 'splinefile', 'distance',
               'poni1', 'poni2', 'rot1', 'rot2', 'rot3', 'wavelength'}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        key, separator, value = line.partition(':')
        key, value = key.strip().lower(), value.strip()
        if not separator or key not in allowed or key in fields:
            raise ValueError('unsupported self-contained PONI geometry')
        fields[key] = value
    if (fields.get('poni_version') != '2' or fields.get('detector', '').lower() != 'detector'
            or Path(fields['detector']).is_file() or fields.get('splinefile', 'none').lower() != 'none'):
        raise ValueError('unsupported self-contained PONI detector or spline file')
    config = json.loads(fields.get('detector_config', '{}'))
    if not isinstance(config, dict) or set(config) - {'pixel1', 'pixel2', 'max_shape'}:
        raise ValueError('detector configuration must contain only inline numeric values')
    for key, value in config.items():
        if key == 'max_shape':
            valid = isinstance(value, list) and len(value) == 2 and all(type(n) is int and n > 0 for n in value)
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        if not valid:
            raise ValueError('detector configuration must contain only inline numeric values')


def decode(path):
    import fabio
    with fabio.open(str(path)) as detector:
        # The pinned EDF reader parses headers first and opens external payloads on .data.
        if any(key.lower() == 'edf_binaryfilename' for key in detector.header):
            raise ValueError('external EDF binary files are unsupported')
        return detector.data


def execute(request):
    root, output = Path(request['inputs']), Path(request['output'])
    import numpy as np
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if request['operation'] == 'decode':
            data = decode(checked(root, request['file']))
            if data.ndim != 2 or not np.isfinite(data).all():
                raise ValueError('detector array must be finite and two-dimensional')
            np.savez_compressed(output, intensity=data)
            details = {'shape': list(data.shape)}
        else:
            import pygix
            case, args = request['case'], request['parameters']
            paths = {name: checked(root, case[name]) for name in ('raw', 'geometry', 'mask')}
            validate_geometry(paths['geometry'])
            raw = decode(paths['raw'])
            mask = decode(paths['mask'])
            if raw.ndim != 2 or raw.shape != mask.shape or not np.isfinite(raw).all() or not np.isfinite(mask).all():
                raise ValueError('raw/mask must be finite, matching two-dimensional arrays')
            pg = pygix.Transform()
            pg.load(str(paths['geometry']))
            supplied_distance = float(pg.dist)
            pg.sample_orientation = args['sample_orientation']
            pg.incident_angle = args['incident_angle']
            pg.tilt_angle = args['tilt_angle']
            pg.dist *= args['distance_scale']
            result = pg.transform_reciprocal(raw, method='lut', correctSolidAngle=True,
                                            mask=mask if args['mask_policy'] == 'supplied' else None, all=True)
            np.savez_compressed(output, intensity=result['I'], qr=result['in_plane'], qz=result['out_plane'], count=result['count'])
            details = {'shape': list(result['I'].shape), 'executed_geometry': {
                'distance_m': float(pg.dist), 'supplied_distance_m': supplied_distance,
                'wavelength_m': float(pg.wavelength), 'poni1_m': float(pg.poni1), 'poni2_m': float(pg.poni2),
                'rot1_rad': float(pg.rot1), 'rot2_rad': float(pg.rot2), 'rot3_rad': float(pg.rot3),
                'incident_angle': float(pg.incident_angle), 'sample_orientation': int(pg.sample_orientation),
                'tilt_angle': float(pg.tilt_angle)}, 'method': 'lut', 'correctSolidAngle': True,
                'mask_policy': args['mask_policy']}
        # Recheck the input identities after the operation as well.
        if request['operation'] == 'decode':
            checked(root, request['file'])
        else:
            for name in ('raw', 'geometry', 'mask'):
                checked(root, case[name])
    return {'runtime': runtime(), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'warnings': [str(w.message) for w in captured], **details}


def main():
    if sys.argv[1:] == ['--probe']:
        print(json.dumps(runtime()))
        return
    output, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        result = execute(json.loads(Path(sys.argv[1]).read_text()))
    result['diagnostics'] = {'stdout': output.getvalue(), 'stderr': errors.getvalue()}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
