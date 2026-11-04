"""Collector integration example: Tag scan -> explicit choice -> visual resume."""
import argparse
import json
from pathlib import Path
import sys
import time

from .api import ROOT, run_recording


def add_arguments(parser):
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New isolated RUN directory')
    parser.add_argument('--visual-assist', choices=['ask', 'yes', 'no'], default='ask',
                        help='ask waits for explicit input; yes/no are explicit frontend decisions')
    parser.add_argument('--sam-model', choices=['efficientsam3', 'sam3'], default='efficientsam3')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pipeline.json')
    parser.add_argument('--threads', type=int, choices=range(1, 65), default=6)
    parser.add_argument('--start-s', type=float)
    parser.add_argument('--end-s', type=float)
    parser.add_argument('--resume', action='store_true', help='Resume this example after a previous Tag scan')
    parser.add_argument('--mcap-depth-mode', choices=['raw', 'metadata'], default='metadata',
                        help='MCAP canonical view depth policy; metadata is the lightweight Tag/HumanV2 default for this example')


def ask_visual_assist(model, read_input=input):
    """No timeout, blank answer, EOF or unattended stdin can silently opt in."""
    while True:
        try:
            answer = read_input(f'是否启用视觉辅助（{model}，Intel GPU）？[y/N] ').strip().lower()
        except EOFError:
            return False, 'stdin_eof_no_consent'
        if answer in ('y', 'yes', '是'):
            return True, 'interactive_yes'
        if answer in ('', 'n', 'no', '否'):
            return False, 'interactive_no'
        print('请输入 y 或 n。')


def print_counts(checkpoint):
    print('\nAprilTag 初检完成（“缺失”为 Tag 定位无效，不是 RGB 采集丢帧）', flush=True)
    for key, label in [('full_timeline', '完整时间轴'), ('selected_video_range', '选定成片区间')]:
        count = checkpoint[key]
        print(f'{label}：{count["frames"]} 行；'
              f'左手缺 {count["sides"]["left"]["tag_missing"]}，'
              f'右手缺 {count["sides"]["right"]["tag_missing"]}；'
              f'任一手缺 {count["any_hand_tag_missing"]}，'
              f'双手同时缺 {count["both_hands_tag_missing"]}；'
              f'重复相机样本 {count["repeated_camera_samples"]}。', flush=True)


def print_artifacts(paths):
    print('\n中间产物与交付路径（不存在的路径是后续计划产物）：', flush=True)
    for key, value in paths.items():
        state = '已生成' if Path(value).exists() else '尚未生成'
        print(f'  {key} [{state}]: {value}', flush=True)


def run_guided(args, read_input=input):
    """明确的三次调用边界，供同事的采集平台/网页按钮接入参考。

    1. run_recording(stop_after_apriltag=True)：读取/校验源、Tag→底面→线圈，
       生成相机光学系轨迹与缺失统计，尚不推理 SAM/HaWoR、不生成成片。
    2. 展示统计并等待用户决定；UI 可用 yes/no 对应用户实际选择。
    3. 同一参数 run_recording(resume=True, visual_assist=enabled)：复用 01/02，
       始终完成 HumanV2→触觉→渲染；仅 enabled 时增加分割/MANO/根融合。
       选择“不使用”也完成 Tag-only 处理，绝不降级为 flow。
    """
    from qn_smith.artifacts import now, sha256, write_json
    output = args.output.expanduser().resolve()
    if (output/'result.json').exists():
        raise ValueError('This RUN is already complete; inspect it or choose a new output directory')
    # 先冻结后续模型选择和设备；确认前后须保持一致以通过 request 身份校验。
    # cpu 仅是张量/预后处理设备；SAM 神经网络与 HaWoR 空间骨干明确使用 Intel GPU。
    options = dict(config=args.config, visual='full', device='cpu', threads=args.threads,
                   start_s=args.start_s, end_s=args.end_s, sam_backend='openvino',
                   sam_model=args.sam_model, hawor_backend='openvino',
                   openvino_model=ROOT/'weights/openvino/hawor-backbone-v2/model.xml',
                   openvino_device='GPU', mcap_depth_mode=getattr(args, 'mcap_depth_mode', 'metadata'))
    # 调用 1：只完成初检。即使 visual='full'，暂停不等于完整视觉已完成。
    checkpoint = run_recording(args.input, output, **options,
                               stop_after_apriltag=True, resume=args.resume)
    print_counts(checkpoint)
    print_artifacts(checkpoint['artifacts'])
    print('\n启用视觉辅助后：Tag 无效帧以视觉根 SE3 候选补充原 HumanV2 骨骼映射；'
          '不替换局部手型，不回写原始 Tag 观测。融合 NPZ 中逐帧保存 valid/source；'
          '证据不足或冲突保留 NaN。', flush=True)
    if args.sam_model == 'efficientsam3':
        print('\n首选 EfficientSAM3（速度优先示例）；仍为 experimental，数值验收未通过。'
              '已测缺 Tag 恢复少于 SAM3；可用 --sam-model sam3 显式比较。', flush=True)
    # 调用 2：请求选择。非交互 ask 遇到 EOF 时保留初检，绝不默认批准。
    if args.visual_assist == 'ask':
        enabled, reason = ask_visual_assist(args.sam_model, read_input)
    else:
        enabled = args.visual_assist == 'yes'
        reason = 'explicit_argument_'+args.visual_assist
    decision = dict(schema='qn_ego.visual_assist_decision.v1', at=now(), enabled=enabled,
                    reason=reason, model=args.sam_model, argv=sys.argv,
                    checkpoint_sha256=sha256(output/'apriltag-checkpoint.json'),
                    input=checkpoint['input'], training_ready=False)
    # 每次请求独立保存，拒绝后恢复也不会覆盖上一轮决定或旧实验。
    write_json(output/'provenance'/f'visual-assist-decision-{time.time_ns()}.json', decision)
    if not enabled:
        print('\n不启动视觉推理；继续 Tag-only 骨骼、触觉与视频，缺 Tag 保留缺测。', flush=True)
    # 调用 3：相同 RUN + 相同 recipe；Journal 校验实际 argv/源码/产物 SHA 后跳过 Tag。
    result = run_recording(args.input, output, **options, resume=True, visual_assist=enabled)
    paths = dict(checkpoint['artifacts'])
    if not enabled:
        folder = output/'mapped/tag-only'/args.input.stem
        paths['fused_skeleton'] = str(folder/'fused-humanv2.npz')
        paths['fusion_report'] = str(folder/'fusion-report.json')
    print_artifacts(paths)
    verification = json.loads((output/'verification.json').read_text())
    if not verification['passed']:
        raise RuntimeError('Output verification did not pass')
    print('\n全链处理与文件校验完成；training_ready=false，缺测/冲突仍按规则保留。', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_guided(parser.parse_args(argv))
