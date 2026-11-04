# Model assets

The local and mini-PC deployment keep models as real files in this directory. Git stores only this README and `manifest.json`; it does not redistribute the ~6.7 GB base checkpoints. A Git clone is a **source checkout**, not an offline deployment bundle.

Acquire SAM3 from its official project under the SAM License, HaWoR from its official project under its published model terms, and MANO from its official authorized distribution. Preserve the paths and SHA256 listed in `manifest.json`. See `../THIRD_PARTY_NOTICES.md` for terms and public-release blockers. The historical hashes identify the tested artifacts; availability or a mirror is not proof of permission.

From an authorized project-layout bundle, copy and validate with:

```bash
python3 scripts/install_assets.py --bundle-root /path/to/authorized-bundle --check
python3 scripts/install_assets.py --bundle-root /path/to/authorized-bundle
```

`--check` validates without copying. Existing matching files are reused; mismatches fail. The installer refuses symlinks, traversal and overwrites. Checkpoints are not downloaded automatically. On mini PC, additionally prepare the selected OpenVINO IR and validation receipts using the provided export tools and deployment docs; these optional models are not in the base manifest. HumanV2 Linux x86_64 runtime libraries are bundled under third_party/humanv2 (see its platform and license requirements). Rendering font and FFmpeg remain separate deployment requirements, not model weights.

Official model download links, exact destination paths, optional Tag-only usage, and SAM3/EfficientSAM3 selection are in [the main README](../README.md#可选视觉纠正模型下载与使用). Tag-only does not require any of these visual checkpoints. Complete vision needs the chosen segmenter plus HaWoR and authorized MANO resources. EfficientSAM3 EV-M is currently OpenVINO GPU only; its optional IR/checkpoint is separate from the base manifest.
