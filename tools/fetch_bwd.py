import os
import urllib.request

BASE = "/home/cq/depthart_test/plugin/vendor"
DA = "https://cdn.jsdelivr.net/gh/xuefeng-cvr/DepthART@main/deploy/shared/selective_scan/vendor/mamba_ssm_csrc/selective_scan/cusoflex/"

for name in ["selective_scan_bwd_kernel_oflex.cuh", "selective_scan_core_bwd.cu"]:
    dst = os.path.join(BASE, name)
    try:
        urllib.request.urlretrieve(DA + name, dst)
        print("ok  ", name, os.path.getsize(dst))
    except Exception as e:
        print("FAIL", name, repr(e)[:80])
