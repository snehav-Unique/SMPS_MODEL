"""Colab asset setup without depending on Git LFS installation."""
import hashlib
import os
from pathlib import Path
import urllib.request
import zipfile

project = Path('/content/SMPS_FINAL')
project.mkdir(parents=True, exist_ok=True)
if not (project/'weights/ensemble_config.json').exists():
    archive = Path('/content/smps_source.zip')
    print('[SETUP] Downloading public project source', flush=True)
    urllib.request.urlretrieve('https://codeload.github.com/snehav-Unique/SMPS_MODEL/zip/refs/heads/main', archive)
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
for relative in assets:
    path = project/relative
    text = path.read_text() if path.stat().st_size < 1024 else ''
    if text.startswith('version https://git-lfs.github.com/spec/v1'):
        expected_hash = next(line.split('sha256:')[1] for line in text.splitlines() if line.startswith('oid '))
        expected_size = int(next(line.split()[1] for line in text.splitlines() if line.startswith('size ')))
        partial = path.with_suffix(path.suffix+'.download')
        print('[DOWNLOAD]', relative, f'({expected_size/1048576:.1f} MiB)', flush=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen('https://media.githubusercontent.com/media/snehav-Unique/SMPS_MODEL/main/'+relative, timeout=120) as response, partial.open('wb') as output:
            while True:
                chunk = response.read(4*1024*1024)
                if not chunk: break
                output.write(chunk)
                digest.update(chunk)
        assert partial.stat().st_size == expected_size, 'Asset size mismatch: '+relative
        assert digest.hexdigest() == expected_hash, 'Asset hash mismatch: '+relative
        partial.replace(path)
        print('[VERIFIED]', relative, flush=True)
    else:
        print('[EXISTS]', relative, path.stat().st_size, 'bytes', flush=True)
print('[SETUP COMPLETE] Official dataset and all five checkpoints are available.', flush=True)
