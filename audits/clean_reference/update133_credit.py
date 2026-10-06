#!/usr/bin/env python3
# Protocol implementation scaffold. Fail closed until exact saved physical-state
# anchors from the update133 evidence are resolved and validated.
import argparse, json, pathlib, sys
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args()
root=pathlib.Path(a.input); out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
files=[str(x.relative_to(root)) for x in root.rglob("*") if x.is_file()]
(out/"inventory.json").write_text(json.dumps({"files":files},indent=2))
required=[x for x in files if "update" in x.lower() or "batch" in x.lower() or "state" in x.lower()]
(out/"preflight.json").write_text(json.dumps({"candidate_files":required},indent=2))
print("Exact conditional-credit execution intentionally fails closed until saved physical-state anchors are identified from the retained evidence.", file=sys.stderr)
sys.exit(2)
