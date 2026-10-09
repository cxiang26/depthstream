path = "/home/cq/.bashrc"
line = 'export LD_LIBRARY_PATH=/home/cq/Projects/TensorRT-10.8.0.43/lib:$LD_LIBRARY_PATH\n'

with open(path) as f:
    lines = f.readlines()

replaced = False
for i, l in enumerate(lines):
    if "TensorRT-10.8.0.43/lib" in l:
        lines[i] = line
        replaced = True

with open(path, "w") as f:
    f.writelines(lines)

print("replaced:", replaced)
print("current:", [l.rstrip() for l in lines if "TensorRT" in l])
