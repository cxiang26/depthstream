"""One-off: patch hardcoded workspace paths after repo reorganization."""
import os

OLD = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper"
NEW = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/depthstream-paper"
ROOT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/depthstream-paper"

for dirpath, _, files in os.walk(ROOT):
    if "__pycache__" in dirpath or "figs" in dirpath:
        continue
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        with open(p, encoding="utf-8") as fh:
            src = fh.read()
        if OLD in src:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(src.replace(OLD, NEW))
            print("patched", os.path.relpath(p, ROOT))
print("done")
