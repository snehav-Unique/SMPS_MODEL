"""Lightweight integrity checks for the generated delivery artifacts."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import ast
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from pypdf import PdfReader

root=Path(__file__).resolve().parent.parent
notebook=json.loads((root/'notebooks/SMPS_Final_Five_Model_Verification.ipynb').read_text(encoding='utf-8'))
count=0
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        source='\n'.join(line for line in source.splitlines() if not line.startswith('%'))
        ast.parse(source)
        count+=1
assert len(PdfReader(root/'docs/reports/final_technical_report.pdf').pages)>=4
page=(root/'presentation/results.html').read_text(encoding='utf-8')
assert 'What to say' not in page and '<pre' not in page and 'FIVE-MODEL ENSEMBLE' in page
assert page.count('data:image/png;base64,')>=9
embedded=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'scripts=json.loads(' in ''.join(c['source']))
namespace={}
# Evaluate only the serialized literal used as the json.loads argument.
tree=ast.parse(embedded)
assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scripts')
scripts=json.loads(ast.literal_eval(assignment.value.args[0]))
for name,source in scripts.items():
    assert source==(root/name).read_text(encoding='utf-8'),f'Stale notebook script: {name}'

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

manifest=json.loads((root/'docs/REORGANIZATION_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    assert (root/entry['current']).is_file(),f"Missing original file: {entry['original']}"
    if Path(entry['original']).suffix.lower() in {'.pth','.pt','.h5','.keras','.onnx','.zip','.mp4','.pdf','.csv','.html'}:
        assert digest(root/entry['current'])==entry['original_sha256'],f"Changed original asset: {entry['current']}"
evidence=json.loads((root/'outputs/final_verification/verification.json').read_text(encoding='utf-8'))
assert evidence['weights_to_csv_status']=='passed'
assert digest(root/'inference_script.py')==evidence['inference_sha256']
config=json.loads((root/'weights/ensemble_config.json').read_text(encoding='utf-8'))
assert len(config['models'])==5 and abs(sum(m['weight'] for m in config['models'])-1)<1e-8
for model in config['models']:
    assert digest(root/'weights'/model['file'])==evidence['weight_sha256'][model['file']]
with (root/'predictions.csv').open(newline='') as stream:
    reader=csv.DictReader(stream)
    assert reader.fieldnames==['image_id','label','confidence']
    rows=list(reader)
assert len(rows)==3000 and len({r['image_id'] for r in rows})==3000
assert all(r['label'] in {'0','1'} and 0<=float(r['confidence'])<=1 for r in rows)
with zipfile.ZipFile(root/'datasets/hackathon_dataset.zip') as archive:
    ids={Path(n).stem for n in archive.namelist() if '/test/images/' in n and not n.endswith('/')}
assert {r['image_id'] for r in rows}==ids
with (root/'outputs/final_verification/predictions_run_01.csv').open(newline='') as stream:
    assert rows==list(csv.DictReader(stream)), 'Final CSV differs from recorded replay'
assert sum((root/'weights'/m['file']).stat().st_size for m in config['models'])<5_000_000_000
assert all(run['process_wall_seconds']<3600 for run in evidence['runs'])
with (root/'data/train_split.csv').open(newline='') as stream:train=list(csv.DictReader(stream))
with (root/'data/val_split.csv').open(newline='') as stream:val=list(csv.DictReader(stream))
assert len(train)==5600 and len(val)==1400
assert not {r['image_id'] for r in train}&{r['image_id'] for r in val}
assert not ({r['image_id'] for r in train}|{r['image_id'] for r in val})&ids
with (root/'outputs/final_validation/statistics.csv').open(newline='') as stream:statistics=list(csv.DictReader(stream))
for stat in statistics:
    with (root/'outputs/final_validation'/('predictions_'+stat['condition']+'.csv')).open(newline='') as stream:
        sample=list(csv.DictReader(stream))
    assert len(sample)==200 and {r['image_id'] for r in sample}<={r['image_id'] for r in val}
    positive=[float(r['confidence']) for r in sample if int(r['label'])==1]
    negative=[float(r['confidence']) for r in sample if int(r['label'])==0]
    auc=sum((p>n)+0.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
    assert abs(auc-float(stat['auc']))<1e-10,(stat['condition'],auc)
# Git LF normalization changes the file hash, but not prediction values.
raw=(root/'predictions.csv').read_bytes().replace(b'\r\n',b'\n')
assert evidence['reference_sha256'] in {hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\n',b'\r\n')).hexdigest()}
for path in list((root/'scripts').glob('*.py'))+list((root/'src').glob('*.py'))+list(root.glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
for name in ['README.md','docs/SUBMISSION.md']:
    path=root/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent/target.replace('%20',' ')).exists(),(name,target)
setup=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'project = Path(' in ''.join(c['source']))
assert (root/'scripts/colab_setup.py').read_text(encoding='utf-8') in setup,'Stale inline Colab setup'
assert '__file__' not in setup,'Inline Colab setup cannot use __file__'
assert {'scripts/robustness_test.py','scripts/render_dashboard.py'}<=scripts.keys()
print(f'PASS: all {len(manifest["files"])} original files retained; model/asset hashes, 3000 official IDs, replay values, Python syntax, links, report length and {count} notebook code cells checked. No fresh GPU execution claimed.')
