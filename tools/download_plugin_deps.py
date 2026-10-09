import os
import urllib.request

BASE = "/home/cq/depthart_test/plugin"
DA = "https://cdn.jsdelivr.net/gh/xuefeng-cvr/DepthART@main/deploy/shared/selective_scan/"
TRT = "https://cdn.jsdelivr.net/gh/NVIDIA/TensorRT@v11.3/include/"

files = {
    DA + "vendor/mamba_ssm_csrc/selective_scan/selective_scan.h": "vendor/selective_scan.h",
    DA + "vendor/mamba_ssm_csrc/selective_scan/selective_scan_common.h": "vendor/selective_scan_common.h",
    DA + "vendor/mamba_ssm_csrc/selective_scan/static_switch.h": "vendor/static_switch.h",
    DA + "vendor/mamba_ssm_csrc/selective_scan/cub_extra.cuh": "vendor/cub_extra.cuh",
    DA + "vendor/mamba_ssm_csrc/selective_scan/reverse_scan.cuh": "vendor/reverse_scan.cuh",
    DA + "vendor/mamba_ssm_csrc/selective_scan/uninitialized_copy.cuh": "vendor/uninitialized_copy.cuh",
    DA + "vendor/mamba_ssm_csrc/selective_scan/cusoflex/selective_scan_oflex.h": "vendor/selective_scan_oflex.h",
    DA + "vendor/mamba_ssm_csrc/selective_scan/cusoflex/selective_scan_oflex.cpp": "vendor/selective_scan_oflex.cpp",
    DA + "vendor/mamba_ssm_csrc/selective_scan/cusoflex/selective_scan_fwd_kernel_oflex.cuh": "vendor/selective_scan_fwd_kernel_oflex.cuh",
    DA + "vendor/mamba_ssm_csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu": "vendor/selective_scan_core_fwd.cu",
    DA + "scripts/build_trt_plugin.sh": "build_trt_plugin.sh",
    TRT + "NvInfer.h": "trt_include/NvInfer.h",
    TRT + "NvInferRuntime.h": "trt_include/NvInferRuntime.h",
    TRT + "NvInferRuntimeCommon.h": "trt_include/NvInferRuntimeCommon.h",
    TRT + "NvInferPlugin.h": "trt_include/NvInferPlugin.h",
    TRT + "NvInferVersion.h": "trt_include/NvInferVersion.h",
    TRT + "NvInferImpl.h": "trt_include/NvInferImpl.h",
}

for url, rel in files.items():
    dst = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst)
        print("ok  ", rel, os.path.getsize(dst))
    except Exception as e:
        print("FAIL", rel, repr(e)[:100])
