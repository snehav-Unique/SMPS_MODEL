"""Lightweight integrity checks for the generated delivery artifacts."""
import ast
import json
from pathlib import Path
from pypdf import PdfReader

root=Path(__file__).resolve().parent
notebook=json.loads((root/'notebooks/SMPS_Final_Five_Model_Verification.ipynb').read_text())
count=0
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        source='\n'.join(line for line in source.splitlines() if not line.startswith('%'))
        ast.parse(source)
        count+=1
assert len(PdfReader(root/'report.pdf').pages)==2
if (root/'report_detailed.pdf').exists():
    assert len(PdfReader(root/'report_detailed.pdf').pages)==4
page=(root/'results.html').read_text(encoding='utf-8')
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
print(f'PASS: {count} notebook cells parse; embedded scripts match; reports are 2/4 pages; offline page has graphs and honest status.')
