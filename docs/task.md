# 项目任务总结

> 维护约束：凡是影响项目现状、已完成改动、下一步优先级或剩余任务清单的开发工作，必须同步更新本文件 `docs/task.md`，避免实现进度与任务文档脱节。

## 一、项目当前现状

本项目目前已经形成了一个基于 `robosuite` 的双臂桌面整理集成基线，用来验证 RoboCasa 风格任务所需的核心链路。当前主流程已经不是“只有框架”，而是已经具备了从感知到执行再到日志记录的基本闭环：

- `vision/` 已经可以运行真实的 GroundingDINO 目标检测，不再只是发布演示假数据。
- `ipc/` 使用规定的 `perception_queue` 完成感知到逻辑的传递。
- `fsm/` 可以消费真实视觉结果，并进入 `PLANNING / CLEARING / GRASPING` 等状态。
- `arm/` 可以在 `robosuite` 中执行抓取和推障动作。
- `logs/` 可以记录运行结果、失败计数以及本轮执行所依赖的数据源。

当前系统已经不再卡在“视觉模型没接上”这个阶段。真实视觉推理已经运行起来，真实检测结果已经进入 FSM，FSM 也已经在驱动抓取尝试。

目前项目的主要瓶颈已经从“视觉不可用”转移到“**双臂抓取执行还不稳定**”。

最近几轮真实运行中已经验证：

- 视觉能够初始化成功，并输出 `status=ready`。
- FSM 能根据真实视觉结果从 `PLANNING` 转入 `GRASPING`。
- 双臂抓取已经开始使用视觉生成的目标点。
- 当抓取失败时，最终状态会正确进入 `RETRY_GRASP`，不会再误判为 `SUCCESS`。
- 失败分类已经与真实行为一致，例如抓取失败会正确计入 `n3`。

## 二、当前主要问题

### 1. 相机外参仍然是占位值

`configs/camera.yaml` 中的 `T_world_cam` 目前仍然是单位阵占位值，不是真实标定结果。

这意味着视觉输出的世界坐标并不是真正可信的物理坐标。目前为了继续推进开发，已经增加了仿真侧的临时修正逻辑，但这只是过渡方案，不是最终解法。

### 2. 双臂抓取执行仍然不稳定

从最近的运行结果看，系统已经能够：

- 检测目标，
- 生成抓取点，
- 驱动双臂接近目标，

但仍然无法稳定地完成夹持和提升。现在的主要问题已经不是“看不到目标”，而是“**双臂在执行阶段无法稳定收敛到合适抓取状态**”。

### 3. 视觉抓取点仍然是启发式方案

当前双臂抓取点生成依赖如下启发式：

- 使用视觉检测得到目标中心，
- 在仿真里对 `x/y` 朝真实物体中心做收缩修正，
- 在仿真里对 `z` 使用真实物体高度做修正，
- 沿固定轴对称生成左右抓取点。

这个方案适合当前 `robosuite` 阶段快速联调，但还不是最终可迁移到 RoboCasa 的正式感知抓取方案。

### 4. `main.py` 仍然偏向集成测试脚本

虽然 `main.py` 已经从最初的大杂烩入口拆成了几个阶段函数，但本质上仍然是一个“集成验证脚本”，而不是彻底清晰的任务调度程序。后续如果继续扩展功能，还需要进一步模块化。

## 三、本轮已经完成的改进

## 1. 真实视觉能力接入

- 新增 `vision/detector.py`，封装 GroundingDINO 检测后端。
- 重写 `vision/perception_loop.py`，实现 `RGB + depth -> detected_objects`。
- 保留并遵守项目要求的 `detected_objects` 结构和 `perception_queue` 协议。
- 让视觉输出进入 FSM，而不是停留在独立演示阶段。

## 2. 运行环境与依赖检查优化

- 增加了对 `torch`、`Pillow`、`transformers` 的显式检查。
- 对缺依赖和版本不兼容给出更明确的错误提示。
- 修复了 GroundingDINO 在不同 `transformers` 版本下后处理参数名不同的问题：
  - 有的版本使用 `box_threshold`
  - 有的版本使用 `threshold`

## 3. Hugging Face 模型加载优化

- 在 `configs/vision.yaml` 中加入 `local_files_only: true`。
- 模型加载优先使用本地缓存，不再每次都去访问 HF Hub。
- 目前每次运行看到的 `Loading weights` 主要是“本地权重重新载入内存”，不是重新下载。

## 4. FSM 与真实视觉联动

- 新增真实视觉消费路径，使 FSM 可以根据真实检测结果进入对应状态。
- 将 demo fallback 与 live perception 路径分开。
- 规划阶段改为在需要时同步执行真实视觉推理，避免 CPU 推理速度慢于异步 worker 超时窗口的问题。

## 5. 主流程结构整理

- 将 `main.py` 拆分为多个阶段函数：
  - `initialize_vision_system`
  - `run_planning_phase`
  - `run_clearing_phase`
  - `run_grasp_phase`
- 主流程结构比之前清晰很多，便于继续演进。

## 6. 成功 / 失败判定修正

- 修复了“进入抓取阶段就被误判成功”的问题。
- 现在运行成功与否基于真实抓取 / 放置结果，而不是仅仅基于 FSM 是否走到了 `GRASPING`。
- 抓取失败会正确回到 `RETRY_GRASP / FAILED` 路径。

## 7. 双臂视觉抓取接入

- 新增双臂视觉抓取点生成。
- 增加仿真高度修正：视觉 `z` 使用真实物体高度进行校正。
- 增加仿真 XY 修正：视觉 `x/y` 向真实物体中心做一定收缩。
- 支持固定抓取轴展开，不再只依赖当前双臂连线方向。
- 将双臂横向偏移调整为更保守的默认值。

## 8. 日志增强

- 扩展了 `execution_summary`。
- 日志中可记录：
  - 规划使用的数据源
  - 清障使用的数据源
  - 抓取使用的数据源
  - 推障使用的数据源
  - 抓取模式
- 同时加入了以下信息：
  - `vision_target_pos`
  - `grasp_targets`
  - `object_pos_before_grasp`
  - `object_pos_after_grasp`

这些信息已经足够支持下一阶段的调参与误差分析。

## 四、下一步最优先的改进方向

### 优先级 1：提高双臂执行收敛能力

这是当前最重要的问题。

从最近几次运行看，系统已经能看到目标、生成抓取点并驱动双臂接近，但双臂仍然无法稳定完成有效夹持。下一步应优先处理：

- 双臂 waypoint 到位能力不足的问题；
- 某一侧机械臂经常未真正到达目标点的问题；
- 双臂闭合前缺少更稳定的对齐过程的问题。

建议方向：

- 增加双臂 waypoint 的最大步数；
- 增强双臂接近阶段的中间过渡点；
- 在闭合夹爪前输出每只手的最终误差；
- 重新调节 `MOVE_GAIN`、双臂步进上限与容差。

### 优先级 2：改进双臂预抓取分阶段动作

当前双臂从当前位置到抓取点的动作还是偏直接。后续应增加更稳的分阶段过程：

- 高位预抓取点
- 中间对齐点
- 最终抓取点
- 闭合前二次微调

### 优先级 3：完成真实外参标定

当前仿真修正方案虽然可用，但不能替代真实标定。应尽快完成：

- `T_world_cam` 标定
- 去掉对真实物体中心的仿真修正依赖
- 用仿真真值验证视觉世界坐标误差

### 优先级 4：增强视觉抓取语义

当前抓取点是围绕中心展开，后续更好的方案应当估计：

- 物体长轴 / 对称轴
- 左右可抓边缘
- 与抓取姿态相关的目标点

## 五、后续待完成任务清单

## 1. 近期任务

1. 提高双臂 waypoint 到位稳定性。
2. 为双臂抓取增加分阶段接近路径。
3. 在双臂闭合前记录每只手与目标点的最终误差。
4. 根据新日志内容重新调节：
   - `dual_grasp_lateral_offset`
   - `dual_grasp_height_offset`
   - `sim_xy_correction_alpha`
   - 双臂控制参数与容差
5. 重新进行多轮实验，对比修改前后的抓取成功率。

## 2. 中期任务

1. 完成 `T_world_cam` 的真实标定。
2. 逐步减少并最终去掉仿真侧的抓取点修正逻辑。
3. 继续完善基于真实视觉的障碍物清理策略。
4. 将更多逻辑从 `main.py` 中拆出，形成更清晰的任务调度结构。
5. 补充测试，包括：
   - detector 输出结构测试
   - FSM 真实感知状态切换测试
   - run log 内容测试
   - 成功 / 失败状态正确性测试

## 3. 后期任务

1. 以环境封装层为边界，从 `robosuite` 迁移到 RoboCasa。
2. 在迁移过程中保持 `vision/`、`ipc/`、`fsm/`、`logs/` 接口稳定。
3. 扩展实验记录能力，用于最终报告：
   - 抓取目标诊断数据
   - 双臂收敛误差
   - 感知延迟
   - 不同随机种子的成功率

## 六、结论

项目目前已经从“核心链路未打通”进入到了“核心链路已运行，但双臂抓取执行质量不足”的阶段。这是明确的阶段性进展。

当前系统已经能够：

- 运行真实视觉推理；
- 通过 `perception_queue` 传递真实感知结果；
- 由真实视觉驱动 FSM 状态切换；
- 生成双臂视觉抓取目标；
- 执行抓取和推障动作；
- 输出较完整的运行日志与执行摘要。

下一阶段的工作重点不应该再主要放在“继续补视觉基础设施”，而应该集中到 **双臂控制收敛、抓取动作分阶段设计、以及真实相机标定** 上。

简而言之：

- 感知链路已经基本打通；
- 逻辑链路已经基本打通；
- 当前最大的工程问题是执行层稳定性。
## 2026-04-26 Execution Layer Update

- Implemented a staged dual-arm grasp approach in `arm/controller.py`: high pre-grasp, mid alignment, short final descent, gripper settle, then lift.
- Added adaptive near-target waypoint stepping to reduce overshoot during the last centimeters of approach.
- Added dual-arm alignment diagnostics to `execution_summary` so runs now record per-arm Cartesian target error before the grasp sequence.
- Immediate next verification: run repeated dual-arm transfer trials and compare `dual_arm_alignment_errors`, grasp completion rate, and post-lift object height against logs from 2026-04-25.
- Added stage-level dual-arm execution diagnostics: each attempt now records `pre_grasp`, `align`, `final_descent`, `grasp_pose`, `lift`, and transfer-stage waypoint errors plus the first `failed_stage`.
- Dual-arm retry attempts now refresh perception before re-grasping, and grasp failure routing distinguishes execution drift from physical slip for cleaner `n2/n3` statistics.
- Push failure is now recorded in the execution summary and counted in the final run success decision.
- Safe arm retract now returns a boolean result that is recorded in `execution_summary` and included in the final run success decision.
- Dual-arm pre-grasp now has a `transit` fallback with single-arm compensation to reduce early stage failure on long approach paths.
- Terminated `robosuite` episodes now short-circuit wrapper steps and retry loops instead of raising during late-stage checks.
- Post-grasp home/push/retract now skip entirely once the episode has terminated, so the log separates terminal task failure from post-failure cleanup.
- Corrected dual-arm target semantics: vision-derived dual targets are now treated as final grasp points, while observation handle targets still receive the handle-to-grasp z offset inside the controller. This removes the previous double application of `dual_grasp_height_offset` / `HANDLE_GRASP_Z_OFFSET`.
- Changed dual-arm approach order from `pre_grasp` with `transit` fallback to an explicit `transit -> pre_grasp -> align -> final_descent -> grasp_pose` sequence, so long-distance motion stays high before descending near the object.
- Added `transit` to execution-drift failure classification for cleaner `n2` accounting when the high approach stage cannot converge.
- Added contact-aware final descent handling: if both arms are XY-aligned and remain only slightly above the grasp point, execution now proceeds to gripper close instead of forcing a lower waypoint that can terminate the episode.
- Increased the high `transit` stage step budget so long-distance approach failures are less likely to be caused by premature timeout rather than true infeasibility.
- Fixed final run accounting so post-terminal push failures are still reflected in `n2` and the console prints both FSM final state and overall `Run success`.
- Dual-arm grasp target selection now prefers robosuite handle observations before falling back to vision-derived symmetric grasp points, so Task 1 starts with pot-handle grasping when handle poses are available.
- Dual-arm transfer after lift now moves in short synchronized segments and checks object height after each segment to detect transfer slip before the pot is dragged or dropped.
- Transfer-stage failures are now categorized as `physical_slip` because they occur after gripper closure and successful lift rather than during free-space approach.
- Added a clearance-based dual-arm release sequence: partial open, retract away from the object center with a small lift, then fully open. This mitigates pot-handle snagging caused by fixed gripper orientation during object rotation.
- Replaced the fixed-orientation limitation with closed-loop wrist-yaw commands derived from the current pot handle axis. Dual-arm approach, lift, carry, low placement, and release now use `action[3:6]` / `action[10:13]` instead of leaving EEF rotation at zero.
- Added a transfer-specific waypoint tolerance so segmented carry stages do not fail on small residual tracking error when both arms remain close and the object is still lifted.
- Added minimal wrist-yaw compliance during segmented carry and release (`action[5]` / `action[12]`) plus a short unsnag wiggle retry for transfer stages that appear to bind on a rotated pot handle.
- Push is again part of final evaluation via `REQUIRE_PUSH_TEST_SUCCESS = True`; push failures count as execution drift and make the integration run fail.
- Changed dual-arm release from an in-air open to a surface-aware place sequence: carry high, descend until the object is near the configured place-table height, partially open, retract outward/upward, then fully open.
- Updated push execution to approach open from above, descend to a contact waypoint, briefly close for a stable pushing surface, then push through the target direction.
