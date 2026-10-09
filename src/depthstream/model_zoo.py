"""Download the student ONNX weights from the official DepthART release.

Weights are NOT distributed with this package: they inherit CC BY-NC 4.0
terms from DepthART / Depth Anything V2 (non-commercial use). By running
this helper you download them from the official HuggingFace repository and
accept those terms.
"""
import os

REPO_ID = "Fengxue93/DepthART"
FILENAME = "onnx/relative_mnv4m_slim_spf_224_default.onnx"
NOTICE = (
    "Downloading DepthART student weights (CC BY-NC 4.0, non-commercial)\n"
    "from the official repository %s (revision: %s).\n"
    "By continuing you accept the upstream license terms.")


def default_cache_dir():
    return os.environ.get("DEPTHSTREAM_CACHE",
                          os.path.join(os.path.expanduser("~"),
                                       ".cache", "depthstream"))


def ensure_student_onnx(cache_dir=None) -> str:
    """Return a local path to the student ONNX, downloading on first use."""
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "depthart_mnv4m_slim_spf_224.onnx")
    if os.path.exists(local) and os.path.getsize(local) > 10**6:
        return local

    print(NOTICE % (REPO_ID, "main"))
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    except ImportError:
        import urllib.request
        base = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = f"{base}/{REPO_ID}/resolve/main/{FILENAME}"
        path = os.path.join(cache_dir, "_download.onnx")
        urllib.request.urlretrieve(url, path)
    with open(local, "wb") as out, open(path, "rb") as src:
        while True:
            chunk = src.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    return local
