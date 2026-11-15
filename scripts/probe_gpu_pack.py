"""Frozen-environment import gate for the GPU runtime pack.

The ONLY faithful way to know whether a staged/installed GPU runtime pack is
self-sufficient is to import it from inside a PyInstaller-frozen process with
the SAME module set as the shipped CPU sidecar. A plain venv python proves
nothing: it has the full stdlib and its own site-packages, so `import torch`
passed there while dying inside the real installed app on stdlib modules the
sidecar's analysis had pruned (pickletools, filecmp, cProfile — v0.15.0,
found live).

Run this against the staged pack BEFORE publishing a gpu-runtime release:

  # 1. Freeze the probe with the sidecar's exact flags (CPU branch of
  #    package_backend_windows.ps1) from the CPU packaging venv:
  .venv-package\\Scripts\\pyinstaller.exe --clean --noconfirm --onefile \
    --name gpu_pack_probe \
    --collect-all onnxruntime --collect-all deep_translator \
    --collect-data melo --collect-all unidic_lite --collect-data pykakasi \
    --hidden-import amadeus.provider_loader \
    --hidden-import amadeus.desktop.server --hidden-import amadeus.web.app \
    --hidden-import amadeus.web.streaming \
    --hidden-import amadeus.providers.llm.gemini_provider \
    --hidden-import amadeus.providers.llm.fallback \
    --hidden-import amadeus.agents.tool_loop \
    --hidden-import amadeus.tools.literature \
    --hidden-import amadeus.tools.desktop_files \
    --hidden-import amadeus.reminders \
    --hidden-import amadeus.providers.tts.melotts_onnx_provider \
    --hidden-import google.generativeai --hidden-import onnxruntime \
    --hidden-import melo.text --hidden-import melo.text.japanese \
    --hidden-import melo.text.symbols --hidden-import unidic_lite \
    --exclude-module tensorflow --exclude-module pandas \
    --exclude-module matplotlib --exclude-module sklearn \
    --exclude-module torch --exclude-module transformers \
    --exclude-module scipy --exclude-module torchvision \
    --distpath dist-desktop\\_probe scripts\\probe_gpu_pack.py

  # 2. Point it at the pack root (the dir holding site-packages/ and sox/):
  dist-desktop\\_probe\\gpu_pack_probe.exe dist-desktop\\gpu-runtime-stage
  dist-desktop\\_probe\\gpu_pack_probe.exe "%APPDATA%\\com.amadeus.kurisu.gpu\\gpu-runtime\\gpu-runtime-v2"

Exit code 0 = every import the Qwen3 voice needs resolves inside the frozen
environment. Anything else = the pack would ship a silent ONNX fallback.
"""

import os
import sys
import traceback

# Pull in the sidecar's real import graph so the frozen module set matches
# the shipped amadeus-backend.exe (no-op when run unfrozen; the freeze step
# is what gives this meaning).
import amadeus.desktop.server
import amadeus.web.app


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gpu_pack_probe <pack-root-dir>", file=sys.stderr)
        return 2
    pack = os.path.abspath(sys.argv[1])
    sp = os.path.join(pack, "site-packages")
    if not os.path.isdir(sp):
        print(f"no site-packages under {pack}", file=sys.stderr)
        return 2

    print("frozen:", getattr(sys, "frozen", False))
    print("python:", sys.version)
    print("pack:", pack)

    # Mirror gpu_runtime.activate() exactly: append (frozen modules win),
    # DLL dirs for torch and sox, sox on PATH.
    sys.path.append(sp)
    if hasattr(os, "add_dll_directory"):
        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
        sox_dir = os.path.join(pack, "sox")
        if os.path.isdir(sox_dir):
            os.add_dll_directory(sox_dir)
    os.environ["PATH"] = (
        os.path.join(pack, "sox") + os.pathsep + os.environ.get("PATH", "")
    )

    checks = [
        (
            "torch",
            "import torch; print('  torch', torch.__version__, 'cuda:', torch.cuda.is_available())",
        ),
        ("qwen_tts", "import qwen_tts"),
        ("faster_qwen3_tts", "from faster_qwen3_tts import FasterQwen3TTS"),
        ("soxr", "import soxr"),
        (
            "sox-binary",
            "import subprocess; subprocess.run(['sox', '--version'], check=True, capture_output=True)",
        ),
    ]
    failures = 0
    for label, stmt in checks:
        try:
            exec(stmt)
            print(f"OK: {label}")
        except BaseException:
            failures += 1
            print(f"FAIL: {label}")
            traceback.print_exc()

    print("RESULT:", "ALL-OK" if failures == 0 else f"{failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
