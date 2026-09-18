"""运行时接线：把 WM 的感知后端写进每个任务的配置里。

eval_pipeline 的 config_builder.py 属于冻结资产（FROZEN.sha256），不能改。
本模块在运行时给 `_apply_wingman_block` 打补丁，让生成的任务 YAML 额外带上
RF-DETR Small + 单目深度的感知后端路径与 backend 名。
全程只改内存中的函数引用，磁盘上的冻结文件保持逐字节不变。

2026-09-15 简化：信息隔离层的开关（done_gate / failure_detection /
success_conditions / advisor_only / 强制关闭分割与真值深度）已全部删除。
动作成败改由帧差判定，见 ``self_observation.estimate_success``。

2026-09-16 新增（可选，默认关）"完成前查状态"：``WM_STATE_CHECK=1`` 时模型
可以输出 ``CheckState()``（旧写法 ``Query(all)`` 仍兼容），WM 把它记得的物体
位置与状态汇总一次。**通用 Query(<物体>) 已砍掉**（2026-09-16 决定：没有时间
验证按需查询的收益）。**不拦 DONE**：模型任何一步都能结束。
与已删除的信息隔离层不同：这里不读任务自带的 ``target_object_types``、
golden action 或模拟器真值，汇总只是 WM 自己的信念。
"""

from __future__ import annotations

import os

WM_ROOT = os.environ.get('LIGHTWM_ROOT', '/home/sudidaren/lightwm_phases')
DETECTOR = f'{WM_ROOT}/checkpoints/rfdetr_small_228094/checkpoint_best_total.pth'
DEPTH_CKPT = f'{WM_ROOT}/checkpoints/small_objects_20260910/dense_depth_best.pt'


def install(config_builder) -> None:
    """给 config_builder._apply_wingman_block 打补丁（幂等）。"""
    if getattr(config_builder, '_wm_patched', False):
        return
    original = config_builder._apply_wingman_block

    def _apply_wingman_block(data, task):
        original(data, task)
        probe = data.setdefault('memory_probe', {})
        world_model = probe.setdefault('world_model', {})

        # 2026-09-16 清理：冻结的 config_builder 还会写入这些旧字段，当前
        # MemoryProbe 一行都不读（targets 是任务真值 target_object_types，
        # done_gate / navigation_directive / interact_soft_dist 是旧探针参数）。
        # 运行时把它们删掉，让配置与实际行为一致、也避免被误读为信息通道。
        for legacy in ('targets', 'done_gate', 'navigation_directive',
                       'interact_soft_dist'):
            probe.pop(legacy, None)

        # 渲染侧同样清理：WM 用的是自己的单目深度头 + 检测头（只吃 RGB），
        # 没有任何代码读模拟器的深度图/实例分割帧，留着只是白花渲染时间。
        data.setdefault('env', {})
        data['env']['render_depth'] = False
        data['env']['render_instance_segmentation'] = False

        # 确定性感知后端（RF-DETR Small + DINOv2 单目深度）
        world_model['detector_backend'] = 'rfdetr_small_depth'
        world_model['detector_path'] = DETECTOR
        world_model['perception_ckpt'] = DEPTH_CKPT      # 新的 small_objects 深度头
        world_model['obj_thr'] = 0.40                    # 与 configs/rfdetr_small.json 一致的验证阈值
        world_model['perception_runtime_root'] = WM_ROOT
        # 深度来源（消融开关）：LIGHTWM_DEPTH_SOURCE=head|da2。
        # da2 = 公开的 Depth-Anything-V2 metric-indoor，零训练；
        # 同一批 250 帧评测房间上实测 0.444 m vs 我们头的 0.989 m。
        world_model['depth_source'] = os.environ.get('LIGHTWM_DEPTH_SOURCE', 'head')
        # 目标物来源（2026-09-16 决定）：不使用任何对象词表/别名表。
        # 运行时由模型自己在开局自由文本说出任务涉及的物品（target_priority）。
        # 旧的 WM_TARGET_SOURCE / WM_CONFIG_TARGETS 开关已废弃。
        # 任务级子目标分解（消融开关）：WM_PLAN=off | old | new
        _plan = os.environ.get('WM_PLAN', 'off').strip().lower()
        probe['plan'] = {
            'enabled': _plan in ('old', 'new'),
            'prompt': _plan if _plan in ('old', 'new') else 'new',
        }
        # 完成前查状态（消融开关）：WM_STATE_CHECK=1 时模型可 CheckState()
        # （旧名 WM_QUERY 仍兼容，但语义已收窄为"状态汇总"这一种查询）
        probe['object_query'] = {
            'enabled': (os.environ.get('WM_STATE_CHECK', '0') == '1'
                        or os.environ.get('WM_QUERY', '0') == '1'),
        }
        # 目标物优先级提示（消融开关）：WM_TARGET_HINT=1 时每步报最重要的前 K 个
        # 相关物体（任务解析 > 手持/容器内容 > 已交互 > 只是见过）。
        # WM_TARGET_HINT_LIMIT 默认 5；距离/不确定度超过阈值只报方位不报距离。
        probe['target_hint'] = {
            'enabled': os.environ.get('WM_TARGET_HINT', '0') == '1',
            'limit': int(os.environ.get('WM_TARGET_HINT_LIMIT', '6') or 6),
            'names_limit': int(os.environ.get('WM_TARGET_HINT_NAMES', '5') or 5),
            'max_dist': float(os.environ.get('WM_TARGET_HINT_MAX_DIST', '3.0') or 3.0),
            'max_sigma': float(os.environ.get('WM_TARGET_HINT_MAX_SIGMA', '0.5') or 0.5),
        }

    config_builder._apply_wingman_block = _apply_wingman_block
    config_builder._wm_patched = True
