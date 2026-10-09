import os
import urllib.request

BASE = "/home/cq/depthart_test/plugin/trt_include"
HEADERS = [
    "NvInfer.h",
    "NvInferImpl.h",
    "NvInferLegacyDims.h",
    "NvInferPlugin.h",
    "NvInferPluginBase.h",
    "NvInferPluginUtils.h",
    "NvInferRuntime.h",
    "NvInferRuntimeBase.h",
    "NvInferRuntimeCommon.h",
    "NvInferRuntimePlugin.h",
    "NvInferVersion.h",
    "NvOnnxConfig.h",
    "NvOnnxParser.h",
]

for h in HEADERS:
    url = f"https://api.github.com/repos/NVIDIA/TensorRT/contents/include/{h}?ref=v11.3"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw"})
    try:
        data = urllib.request.urlopen(req, timeout=60).read()
        with open(os.path.join(BASE, h), "wb") as f:
            f.write(data)
        print("ok  ", h, len(data))
    except Exception as e:
        print("FAIL", h, repr(e)[:80])
