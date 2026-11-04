#!/usr/bin/env python3
"""Mini PC 采集平台集成示例（不启动相机，只处理已关闭的 HDF5）。

推荐统一入口：run_pipeline.py guided-example --input ... --output ...
也可直接运行本文件，两者调用完全相同的函数，不另写一套处理流水线。

完整的、带中文逐步注释的调用代码在 src/qn_ego/guided.py::run_guided：
  run_recording(..., stop_after_apriltag=True)
    -> checkpoint['full_timeline'/'selected_video_range'] 反馈缺失帧数
    -> checkpoint['artifacts'] 提供中间产物绝对路径（区分已生成/计划）
    -> ask_visual_assist('efficientsam3') 等待用户确认
    -> run_recording(..., resume=True, visual_assist=用户选择)
       始终完成骨骼/触觉/渲染；用户同意才执行视觉辅助根定位。

接入 GUI 时，先调公共 API run_recording 暂停，将统计返回前端；
只有用户点“启用视觉辅助”后才能用相同参数 resume。不可从工作线程调用
input() 后把 EOF/超时当作同意。--visual-assist yes/no 是显式选择接口。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from qn_ego.guided import main

if __name__ == '__main__':
    main()
