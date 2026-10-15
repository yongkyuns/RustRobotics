"""Export only original final policies, normalizers, panels and diagonal paths.

Requires the exact FULL epoch-reuse ZIP from report5815f1f. No model is retrained
or reconstructed. No raw input or pickle is executed during export.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from crossover_core import SEEDS, PANELS, MODES, CHECKPOINT, require, write_json

FULL_SHA256 = '5378a475cd4f904f5601327a4dc309c4f9f5dadcc5a558a115ced79211d1e230'
COMPACT_SHA256 = 'ebe5966e10ac6fdf1694734886ff554a93fb2f5b5714492ff158a49af4de3eaf'


def wanted_paths() -> list[str]:
    names = ['parent/results/selection/FROZEN.json']
    for s in SEEDS:
        for p in PANELS:
            names.append(f'parent/results/selection/{s}-{p}.npz')
        for n in (4, 10):
            prefix = f'results/{s}-decay{n}/{CHECKPOINT:07d}/'
            names += [prefix + f for f in ('policy.npz', 'NORMALIZER.json', 'RECEIPT.json')]
            names += [prefix + f'{p}-{m}.npz' for p in PANELS for m in MODES]
    return names


def validate_zip_members(names: list[str]) -> None:
    require(len(names) == len(set(names)), 'Duplicate ZIP entries')
    for name in names:
        p = PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                'Unsafe ZIP path')


def export(archive: Path, output: Path) -> None:
    require(archive.is_file(), f'Missing original epoch-reuse archive: {archive}')
    require(not output.exists(), 'Use a new output directory; no replacement of existing inputs')
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    require(h.hexdigest() == FULL_SHA256, 'Wrong original FULL archive digest; compact alone has no required diagonal paths')
    with zipfile.ZipFile(archive) as z:
        validate_zip_members(z.namelist())
        manifest_bytes = z.read('MANIFEST.json')
        manifest = json.loads(manifest_bytes)
        wanted = wanted_paths()
        missing = [p for p in wanted if p not in manifest or p not in z.namelist()]
        require(not missing, 'Required original files absent: ' + ', '.join(missing))
        # Validate all selected bytes before writing any output.
        payloads = {p: z.read(p) for p in wanted}
        for p, data in payloads.items():
            require(hashlib.sha256(data).hexdigest() == manifest[p], 'Original member hash mismatch: '+p)
        output.mkdir(parents=True)
        for p, data in payloads.items():
            dst = output/p; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
        (output/'SOURCE_MANIFEST.json').write_bytes(manifest_bytes)
        subset = {p: manifest[p] for p in wanted}
        subset['SOURCE_MANIFEST.json'] = hashlib.sha256(manifest_bytes).hexdigest()
        write_json(output/'INPUT_MANIFEST.json', subset)
        write_json(output/'ORIGIN.json', dict(protocol=6201344116,
            source_full_sha256=FULL_SHA256, source_report_commit='5815f1f8221ee752192352b20c44108fb3032f51',
            required_file_count=len(wanted), new_training=False,
            input_manifest_sha256=hashlib.sha256((output/'INPUT_MANIFEST.json').read_bytes()).hexdigest()))
        print(json.dumps({'exported_files': len(wanted), 'directory': str(output)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('epoch_full_zip', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    export(a.epoch_full_zip, a.output)
