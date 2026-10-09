path = "/home/cq/.bashrc"
with open(path) as f:
    lines = f.readlines()
kept = [l for l in lines if "TensorRT-10.8.0.43" not in l]
with open(path, "w") as f:
    f.writelines(kept)
print(f"removed {len(lines) - len(kept)} line(s)")
