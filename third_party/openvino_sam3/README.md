# OpenVINO SAM3 conversion reference

`ov_sam3_helper.py` is copied without edits from Intel's OpenVINO notebooks,
commit `b2aa290400902c713ae0ca727211cdae4d3dfaaa`:
https://github.com/openvinotoolkit/openvino_notebooks/tree/b2aa290400902c713ae0ca727211cdae4d3dfaaa/notebooks/sam3

Upstream labels the notebook experimental. Its Apache-2.0 license is included.
This project uses the real-matrix ViT RoPE conversion from that helper. Our
text-only grounding wrapper invokes the bundled original SAM3 forward method,
including its original empty geometry prompt, rather than approximating the
geometry path with padded boxes. It freezes the existing constant `gloved hand`
text features at conversion. It does not support arbitrary new prompts.

Runtime OpenVINO models must pass same-frame mask/box comparisons on the selected
GPU and match their checkpoint and file hashes. No CPU fallback is provided.
