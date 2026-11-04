# Third-party notices and distribution boundaries

The root BSD 2-Clause License covers original contributions owned by wkz14. It does **not** relicense upstream code, model weights, fonts, calibration data, or native SDKs. Existing copyright headers and upstream licenses take precedence in their respective files. Optional restricted components are obtained separately by the user under their upstream terms; the proprietary HumanV2 binary has its own authorization notice.

| Component | Upstream / local location | Terms and scope |
|---|---|---|
| AprilTag tag family | [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag), `licenses/AprilTag-BSD-2-Clause.txt` | BSD 2-Clause, retain the original University of Michigan attribution. Detection here calls OpenCV ArUco's AprilTag 36h11 dictionary; it does not run an independently compiled AprilTag detector. |
| OpenCV / contrib 4.11 | [OpenCV](https://github.com/opencv/opencv), `licenses/OpenCV*-Apache-2.0.txt` | Apache 2.0; installed dependency. Third-party components within OpenCV retain their respective licenses. |
| SAM3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3), `third_party/sam3/LICENSE` | Custom SAM License (2025-11-19), including code/model terms and restrictions. Not BSD/Apache. Retain upstream notices; obtain weights under upstream access terms. |
| EfficientSAM3 | [SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3), `third_party/efficientsam3` | Own code declared Apache 2.0 by its README (snapshot `licenses/EfficientSAM3-README.md`); nested SAM3 has the SAM License. MobileCLIP and other dependencies keep their own terms. This statement does not establish unrestricted rights for every derived checkpoint. |
| MobileCLIP derived code | [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip), `licenses/MobileCLIP-LICENSE.txt` | MIT for the referenced source code; preserve Apple attribution. Checkpoint/model terms are separate. |
| OpenVINO SAM3 wrapper | `third_party/openvino_sam3/LICENSE` | Apache 2.0 wrapper; conversion does not change source model license. |
| EgoSmith | [egosteer/egosmith](https://github.com/egosteer/egosmith), `licenses/EgoSmith-Apache-2.0.txt` | Optional separate checkout. `scripts/hawor_loader.py` adapts its loader under Apache 2.0; separately obtained dependencies retain their own terms. |
| HaWoR | [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR), `licenses/HaWoR-license.txt` | Obtained separately via EgoSmith's pinned submodule; its README identifies models as CC BY-NC-ND 4.0. Its provenance document treats the obtained source as license-sensitive. No HaWoR source or checkpoint is included in this release. |
| MANO | [official MANO terms](https://mano.is.tue.mpg.de/license.html) | Independent model license, registration/access restrictions. No `.pkl` models are shipped in Git. Obtaining a mirror download does not grant redistribution rights. |
| HumanHandV2 | `third_party/humanv2/native/qnbot_hand` | User confirms this is company-owned code. Bundled native libraries have separate notices. |
| Chumpy | [mattloper/chumpy](https://github.com/mattloper/chumpy) | Optional separate upstream checkout; retain its MIT notice. |
| Noto CJK font | [notofonts/noto-cjk](https://github.com/notofonts/noto-cjk), `licenses/Noto-OFL.txt` | SIL OFL 1.1. The unmodified font is bundled for Chinese dashboard labels; its SIL OFL notice is included. |
| FFmpeg / Intel OpenCL / native dependencies | deployment `runtime/`, HumanV2 runtime notices | HumanV2 dependencies are included with their notices; other deployment runtime tools remain separate. Retain notices for each actual build. |

Python dependencies (PyTorch, OpenVINO, NumPy, SciPy, h5py, Pillow, etc.) retain their own licenses; this file is not a substitute for an inventory of the installed deployment environment. Runtime dependencies and model artifacts must be inventoried separately.

## Public-release checklist

- Confirm any converted HaWoR IR redistribution permissions; CC BY-NC-ND is not an unrestricted open-source grant.
- Verify SAM/EfficientSAM3 checkpoint terms and access requirements per artifact. Publish hashes and acquisition instructions rather than restricted blobs.
- Review native dependency notices, personal calibrations, source recordings and logs separately. Git source export omits the deployment binaries, operator references, recordings and private provenance.
- BSD is the original-code license, **not a claim that the whole optional visual stack is permissive or commercially usable**.
