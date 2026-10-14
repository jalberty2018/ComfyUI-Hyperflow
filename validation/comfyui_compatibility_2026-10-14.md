# v1.3.2: ComfyUI update compatibility checks

No HyperFlow runtime incompatibility was reproduced against the installed
ComfyUI update. This release adds compatibility tests and documents the results;
it does not change sampling behavior or weights.

## Verified

- **101 tests passed**, with no skips or failures. The original 88 tests passed
  before adding the new coverage.
- Both node classes import, register and expose their expected inputs/outputs.
- Native CUDA MiniMax-H3 fixtures run with **bf16 and INT8/convrot** backbone
  weights, in **bypass and merge** modes.
- Full-base two-time conditioning, pruned backbone-only conditioning and
  experimental curve-refit routing complete all eight scheduled forwards.
- Each CUDA case loads, samples, unloads and reloads twice. Outputs match
  bit-for-bit across reloads; the original model attributes are restored.
- Native token refinement, attention/rotary kernels, MLP activation dispatch,
  output heads, ModelPatcher injections and diffusion-model wrappers are used.

The CUDA fixtures are small synthetic models, not full-size generation runs.
These results do not identify an unreported user traceback, nor validate every
custom loader, attention extension, workflow or memory/offload configuration.

## Environment

- Checked on 2026-10-14 using the ComfyUI venv.
- ComfyUI: `7bf5df9e38763dbdd94a0f9c939273019025c2bc` (2026-10-13), version 0.39.0.
- GPU: NVIDIA GeForce RTX 5090.
- PyTorch: `2.10.0+cu130`.
- Comfy Kitchen: `0.2.37`.
- comfy-aimdo: `0.5.6`.
- Full suite: `101 passed, 1 warning in 15.65s`.
- Warning: PyTorch's `pynvml` deprecation warning; not a test failure.

Run from the node-pack directory:

```powershell
& C:\Users\drbaph\Documents\ComfyUI\venv\Scripts\python.exe -m pytest tests -q --tb=short
```

CUDA coverage skips on machines without CUDA. The local 5090 run executed all
12 CUDA cases, covering 192 scheduled model forwards across the reload cycles.

## Relevant upstream change

ComfyUI [fixed bypass LoRAs being skipped by fused operations](https://github.com/Comfy-Org/ComfyUI/commit/d91ed5f5b7fa60fa18464c2ad7c80254da2f0f29)
on October 7. The installed October 13 build includes that fix. An intermediate
build before the fix could behave differently; this is a possible explanation,
not a diagnosis of the reports without their logs.

For a failing installation, capture its exact ComfyUI commit, complete traceback,
base/adapter filenames and the `[hyperflow] ... applied` line. The existing
`--disable-fast-disk` troubleshooting advice remains a diagnostic for loading
stalls, not a confirmed remedy for an unidentified error.

No replacement adapter or curve-fit download is needed for this test/documentation
release.
