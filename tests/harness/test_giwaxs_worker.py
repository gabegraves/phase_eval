"""Self-contained worker inputs must not open unregistered dependencies."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from harness.giwaxs import worker


PONI = ('poni_version: 2\nDetector: Detector\n'
        'Detector_config: {"pixel1": 0.0002, "pixel2": 0.0002, "max_shape": [2048, 2048]}\n'
        'Distance: 0.7\nPoni1: 0.38\nPoni2: 0.06\nRot1: 0\nRot2: 0\nRot3: 0\n'
        'Wavelength: 4.96e-11\n')


def request(tmp_path, geometry=PONI):
    case = {}
    for name, filename, data in [('geometry', 'geometry.poni', geometry.encode()),
                                 ('raw', 'raw.tif', b'raw'), ('mask', 'mask.edf', b'mask')]:
        (tmp_path / filename).write_bytes(data)
        case[name] = {'file': filename, 'sha256': hashlib.sha256(data).hexdigest()}
    return {'inputs': str(tmp_path), 'output': str(tmp_path / 'result.npz'),
            'operation': 'transform', 'case': case, 'parameters': {}}


@pytest.mark.parametrize('geometry', [
    PONI + 'SplineFile: ../unbound.spline\n',
    PONI.replace('Detector: Detector', 'Detector: ../unbound.h5'),
    PONI.replace('Detector: Detector', 'Detector: NexusDetector'),
    PONI.replace('"pixel1": 0.0002', '"splineFile": "../unbound.spline"'),
    PONI.replace('"pixel1": 0.0002', '"pixel1": "../unbound"'),
    json.dumps({'poni': '../unbound.poni'}),
])
def test_transform_rejects_indirect_geometry_before_loading(tmp_path, monkeypatch, geometry):
    monkeypatch.setitem(sys.modules, 'pygix', SimpleNamespace(
        Transform=lambda: pytest.fail('native geometry loader must not run')))
    monkeypatch.setattr(worker, 'decode', lambda _: pytest.fail('decoding must not run'))
    with pytest.raises(ValueError):
        worker.execute(request(tmp_path, geometry))


def test_deposited_geometry_form_and_file_shadow(tmp_path, monkeypatch):
    path = tmp_path / 'geometry.poni'
    path.write_text(PONI)
    worker.validate_geometry(path)
    path.write_text(PONI + 'SplineFile: None\n')
    worker.validate_geometry(path)
    # pyFAI checks file names before its built-in detector registry.
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'Detector').write_bytes(b'unregistered detector file')
    with pytest.raises(ValueError, match='detector'):
        worker.validate_geometry(path)


@pytest.mark.parametrize('operation', ['decode', 'transform'])
def test_external_edf_rejected_before_pixel_access(tmp_path, monkeypatch, operation):
    class Image:
        def __init__(self, path):
            self.header = {'EDF_BinaryFileName': 'unbound.bin'} if path.endswith('.edf') else {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @property
        def data(self):
            assert not self.header, 'external payload must not be read'
            return np.zeros((2, 2))

    monkeypatch.setitem(sys.modules, 'fabio', SimpleNamespace(open=Image))
    monkeypatch.setitem(sys.modules, 'pygix', SimpleNamespace(
        Transform=lambda: pytest.fail('transform must not run')))
    payload = request(tmp_path)
    payload.update(operation=operation, file=payload['case']['mask'])
    with pytest.raises(ValueError, match='external EDF'):
        worker.execute(payload)
