"""Verify this SOFTWARE-CONTROL package; never simulate, train or load pickle."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((ROOT/'MANIFEST.json').read_text())
for name,h in manifest.items():
 p=ROOT/name
 if not p.is_file() or sha(p)!=h:raise AssertionError('Missing/changed payload: '+name)
print('PAYLOADS_OK',len(manifest),flush=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'code'),'-p','test_crossover.py','-v'],check=True)
with tempfile.TemporaryDirectory() as tmp:
 subprocess.run([sys.executable,str(ROOT/'code/analyze.py'),'controls','--reference',str(ROOT/'reference'),
  '--results',str(ROOT/'controls'),'--output',tmp],check=True)
 actual=(Path(tmp)/'CONTROL_ANALYSIS.json').read_bytes();expected=(ROOT/'analysis/CONTROL_ANALYSIS.json').read_bytes()
 if actual!=expected:raise AssertionError('Control report does not reproduce byte-for-byte')
 print('CONTROL_REPORT_BYTE_EXACT',hashlib.sha256(actual).hexdigest())
print('SOFTWARE_CONTROLS_VERIFIED — REGISTERED FOUR-PASS CROSSOVER NOT EXECUTED')
