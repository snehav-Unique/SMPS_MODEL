"""Colab asset setup without depending on Git LFS installation."""


import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

source_revision = os.environ.get('SMPS_SOURCE_REVISION', 'a63df93271c87ca749643a30c4bc188d207f2c31')
project = Path(os.environ.get('SMPS_COLAB_ROOT', '/content/SMPS_FINAL'))
project.mkdir(parents=True, exist_ok=True)
if not (project/'weights/ensemble_config.json').exists():
    archive = Path('/content/smps_source.zip')
    print('[SETUP] Downloading public project source', flush=True)
    urllib.request.urlretrieve('https://codeload.github.com/snehav-Unique/SMPS_MODEL/zip/'+source_revision, archive)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = Path(*Path(entry.filename).parts[1:])
            target = (project/relative).resolve()
            assert target.is_relative_to(project.resolve())
            if not entry.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(entry))
    archive.unlink()
os.chdir(project)
assets = ['weights/efficientnetv2_s_best.pth', 'weights/convnext_tiny_best.pth',
          'weights/swin_tiny_best.pth', 'weights/maxvit_tiny_best.pth',
          'weights/densenet121_best.pth', 'datasets/hackathon_dataset.zip']
expected_assets=json.loads('{"weights/efficientnetv2_s_best.pth": {"sha256": "3e5a3858cd40f22bfb885fd99aaedeff1c049ad720a49755e5e8d15a406d5b6e", "bytes": 81617723}, "weights/convnext_tiny_best.pth": {"sha256": "6165cde373bea423bc3da6e4d24d85f0b13bc44ee0bfcfff9b9f96dedfe18d3e", "bytes": 111355879}, "weights/swin_tiny_best.pth": {"sha256": "c45de824da9aa780f67b3ca597ed7a9352d6b957ce7cd825cb1f3b29019d02f3", "bytes": 110149322}, "weights/maxvit_tiny_best.pth": {"sha256": "10baf3a90f3a717aef75067ab87358c0c07717e827a5c3f2684c76082c56d083", "bytes": 122038277}, "weights/densenet121_best.pth": {"sha256": "683de13e7980c6e1365f877ee58ca81ed43f8e1ed87d0281f2f51f21d19bdc9e", "bytes": 28439894}, "datasets/hackathon_dataset.zip": {"sha256": "3fa8ff24720524a182a4cc449bb56f2ee48a588d04e442a8c3cd93a7089dc08e", "bytes": 133274437}}')
for relative in assets:
    path = project/relative
    text = path.read_text(encoding='utf-8') if path.stat().st_size < 1024 else ''
    if text.startswith('version https://git-lfs.github.com/spec/v1'):
        expected_hash = next(line.split('sha256:')[1] for line in text.splitlines() if line.startswith('oid '))
        expected_size = int(next(line.split()[1] for line in text.splitlines() if line.startswith('size ')))
        partial = path.with_suffix(path.suffix+'.download')
        print('[DOWNLOAD]', relative, f'({expected_size/1048576:.1f} MiB)', flush=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen('https://media.githubusercontent.com/media/snehav-Unique/SMPS_MODEL/'+source_revision+'/'+relative, timeout=120) as response, partial.open('wb') as output:
            while True:
                chunk = response.read(4*1024*1024)
                if not chunk: break
                output.write(chunk)
                digest.update(chunk)
        assert partial.stat().st_size == expected_size, 'Asset size mismatch: '+relative
        assert digest.hexdigest() == expected_hash, 'Asset hash mismatch: '+relative
        partial.replace(path)
        print('[VERIFIED]', relative, flush=True)
    expected=expected_assets[relative]
    assert path.stat().st_size == expected['bytes'], 'Cached asset size mismatch: '+relative
    actual=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            actual.update(chunk)
    assert actual.hexdigest() == expected['sha256'], 'Cached asset hash mismatch: '+relative
    print('[VERIFIED ASSET]', relative, flush=True)
print('[SETUP COMPLETE] Official dataset and all five checkpoints are available.', flush=True)
