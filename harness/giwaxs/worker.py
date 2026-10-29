"""Standalone Python 3.10 worker. Only detector decoding and reciprocal transform."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
from importlib.metadata import version
import json
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


def execute(request):
    root, output = Path(request['inputs']), Path(request['output'])
    import fabio
    import numpy as np
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if request['operation'] == 'decode':
            data = fabio.open(str(checked(root, request['file']))).data
            if data.ndim != 2 or not np.isfinite(data).all():
                raise ValueError('detector array must be finite and two-dimensional')
            np.savez_compressed(output, intensity=data)
            details = {'shape': list(data.shape)}
        else:
            import pygix
            case, args = request['case'], request['parameters']
            paths = {name: checked(root, case[name]) for name in ('raw', 'geometry', 'mask')}
            raw = fabio.open(str(paths['raw'])).data
            mask = fabio.open(str(paths['mask'])).data
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
