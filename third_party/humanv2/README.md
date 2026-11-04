# HumanV2 binary runtime

This directory distributes the compiled HumanV2 solver and required runtime resources.
Private C/C++ implementation, headers, build scripts, internal patches, tests and design documents are intentionally excluded.

- Platform: Linux x86_64, glibc 2.38+ and a compatible system libstdc++ (Ubuntu 24.04 or compatible). Other platforms require a matching authorized binary.
- Executable: `runtime/bin/human_hand_v2_runtime` (Release, stripped).
- Shared dependencies: `runtime/lib/`; the pipeline sets `LD_LIBRARY_PATH` through its native wrappers. For direct CLI use, set it to this directory.
- Resources: `native/qnbot_hand/resources/`; solver options: `native/qnbot_hand/configs/human_hand_v2_options.yaml`.
- Keep the full directory layout when relocating. No compiler or old project checkout is required.
- Supply the recording-specific calibration with the pipeline's calibration configuration; personal calibration files are not distributed.
- Input: five EMF sensor poses. Output: fixed-wrist HumanV2 local skeleton in the existing EMF coordinate convention. Tag-to-bottom-to-coil offsets remain the responsibility of the pipeline and must not be applied twice.
- `runtime-manifest.json` inventories SHA256 and sizes. Dependency notices are under `runtime/third_party_licenses/`.

HumanV2 is a separately licensed binary component, not covered by the repository BSD source license. See `LICENSE.txt` and the root third-party notices.
