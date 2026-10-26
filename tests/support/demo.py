import tarfile
from pathlib import Path

from phaseeval.workbench import cli
from phaseeval.workbench.labels import LabelStore
from phaseeval.workbench.model import ChildAttempt, ImportRun


REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / 'tests/workbench/fixtures/dev006_bgmn'


# ponytail: the bundled receipts cover six Qwen runs; add public receipts for other producers.
class Demo:
    def __init__(self, directory):
        self.directory = directory
        with tarfile.open(REPO / 'launcher/demo/phaseeval-demo.tar.xz') as archive:
            archive.extractall(directory, filter='data')
        self.store = cli.open_store(directory / 'store', directory)
        self.labels = LabelStore(self.store)
        self.roots = {self.store.walk(root)[0][1].case.case_id: root for root in self.store.roots(cli.OWNER)}
        self.runs = {
            attempt.action.harbor.job_name: digest
            for root in self.roots.values() for digest, attempt in self.store.walk(root)
            if isinstance(attempt, ChildAttempt) and isinstance(attempt.action, ImportRun)
        }

    def origin(self, case='DEV006', model='qwen', series='006'):
        return self.runs[f'u6-{case.lower()}-{model}-{series}']

    def import_into(self, directory, origins):
        target = cli.open_store(directory, self.directory)
        blobs = {p.name: p.read_bytes() for p in (self.store.root / 'blobs').iterdir()}
        imported = []
        for origin in origins:
            (_, root), (_, attempt) = self.store.lineage(cli.OWNER, origin)
            target_root = target.load_case(cli.OWNER, root.case, blobs)
            run = target.import_run(cli.IMPORTER, target_root, attempt.action, attempt.outcome.result,
                                    {ref.sha256: self.store.read_blob(ref.sha256) for ref in attempt.outcome.origin.receipts})
            imported.append(run)
        return target, imported
