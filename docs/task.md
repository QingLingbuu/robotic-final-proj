# 项目任务总结

> 维护约束：凡是影响项目现状、已完成改动、下一步优先级或剩余任务清单的开发工作，必须同步更新本文件 `docs/task.md`，避免实现进度与任务文档脱节。

## 一、项目当前现状

本项目目前已经形成了一个基于 `robosuite` 的双臂桌面整理集成基线，用来验证 RoboCasa 风格任务所需的核心链路。当前主流程已经具备从感知到执行再到日志记录的基本闭环：

- `vision/` 已经可以运行真实的 GroundingDINO 目标检测，不再只是发布演示假数据。
- `ipc/` 使用规定的 `perception_queue` 完成感知到逻辑的传递。
- `fsm/` 可以消费真实视觉结果，并进入 `PLANNING / CLEARING / GRASPING` 等状态。
- `arm/` 可以在 `robosuite` 中执行抓取和推障动作。
- `logs/` 可以记录运行结果、失败计数以及本轮执行所依赖的数据源。

当前系统已经不再卡在“视觉模型没接上”这个阶段。真实视觉推理已经运行起来，真实检测结果已经进入 FSM，FSM 也已经在驱动抓取尝试。

目前项目的主要瓶颈已经从“视觉不可用”转移到“**双臂抓取执行还不稳定**”。

最近几轮运行中已经验证：

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

- 检测目标；
- 生成抓取点；
- 驱动双臂接近目标；

但仍然无法稳定地完成夹持、提升、搬运与放置。现在的主要问题已经不是“看不到目标”，而是“**双臂在执行阶段无法稳定收敛到合适抓取状态**”。

### 3. 视觉抓取点仍然是启发式方案

当前双臂抓取点生成依赖如下启发式：

- 使用视觉检测得到目标中心；
- 在仿真里对 `x/y` 朝真实物体中心做收缩修正；
- 在仿真里对 `z` 使用真实物体高度做修正；
- 沿固定轴对称生成左右抓取点。

这个方案适合当前 `robosuite` 阶段快速联调，但还不是最终可迁移到 RoboCasa 的正式感知抓取方案。

### 4. `main.py` 仍然偏向集成测试脚本

虽然 `main.py` 已经从最初的大杂烩入口拆成了几个阶段函数，但本质上仍然是一个“集成验证脚本”，而不是彻底清晰的任务调度程序。后续如果继续扩展功能，还需要进一步模块化。

### 5. 工程完整度仍然不足

当前系统可以演示主链路，但距离“完整项目”仍有差距：

- 自动化测试覆盖不足；
- 依赖与运行环境存在不一致风险；
- README / 配置 / 文档对新成员的可复现支持不够强；
- 评估流程与标准场景还没有完全固化；
- `robosuite -> RoboCasa` 的迁移边界虽然明确，但尚未做成真正可替换的接口层。

### 6. RL 尚未进入可落地阶段

项目当前没有使用 RL 训练，机械臂动作主要依靠：

- 感知或环境观测给出目标点；
- FSM 选择当前阶段；
- `arm/controller.py` 按 waypoint 和误差反馈生成 action；
- `robosuite` 内部控制器执行底层运动。

这条链路是可行的，但如果要接入 RL，目前仍缺少稳定标定、充足轨迹数据、标准评估基线和训练闭环。

## 三、本轮之前已经完成的改进

### 1. 真实视觉能力接入

- 新增 `vision/detector.py`，封装 GroundingDINO 检测后端。
- 重写 `vision/perception_loop.py`，实现 `RGB + depth -> detected_objects`。
- 保留并遵守项目要求的 `detected_objects` 结构和 `perception_queue` 协议。
- 让视觉输出进入 FSM，而不是停留在独立演示阶段。

### 2. 运行环境与依赖检查优化

- 增加了对 `torch`、`Pillow`、`transformers` 的显式检查。
- 对缺依赖和版本不兼容给出更明确的错误提示。
- 修复了 GroundingDINO 在不同 `transformers` 版本下后处理参数名不同的问题：
  - 有的版本使用 `box_threshold`
  - 有的版本使用 `threshold`

### 3. Hugging Face 模型加载优化

- 在 `configs/vision.yaml` 中加入 `local_files_only: true`。
- 模型加载优先使用本地缓存，不再每次都去访问 HF Hub。
- 目前每次运行看到的 `Loading weights` 主要是“本地权重重新载入内存”，不是重新下载。

### 4. FSM 与真实视觉联动

- 新增真实视觉消费路径，使 FSM 可以根据真实检测结果进入对应状态。
- 将 demo fallback 与 live perception 路径分开。
- 规划阶段改为在需要时同步执行真实视觉推理，避免 CPU 推理速度慢于异步 worker 超时窗口的问题。

### 5. 主流程结构整理

- 将 `main.py` 拆分为多个阶段函数：
  - `initialize_vision_system`
  - `run_planning_phase`
  - `run_clearing_phase`
  - `run_grasp_phase`
- 主流程结构比之前清晰很多，便于继续演进。

### 6. 成功 / 失败判定修正

- 修复了“进入抓取阶段就被误判成功”的问题。
- 现在运行成功与否基于真实抓取 / 放置结果，而不是仅仅基于 FSM 是否走到了 `GRASPING`。
- 抓取失败会正确回到 `RETRY_GRASP / FAILED` 路径。

### 7. 双臂视觉抓取接入

- 新增双臂视觉抓取点生成。
- 增加仿真高度修正：视觉 `z` 使用真实物体高度进行校正。
- 增加仿真 XY 修正：视觉 `x/y` 向真实物体中心做一定收缩。
- 支持固定抓取轴展开，不再只依赖当前双臂连线方向。
- 将双臂横向偏移调整为更保守的默认值。

### 8. 日志增强

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

## 四、项目完整化方案

### 1. 架构完整化

目标：把项目从“单入口集成脚本”推进到“模块边界清晰的可维护系统”。

建议：

- 将 `main.py` 继续收缩为启动入口。
- 抽出任务编排层，例如 `task_runner` / `orchestrator`，统一调度 perception、planning、clearing、grasp、place、push。
- 让 `arm/env_wrapper.py` 成为环境层唯一交换边界，为 `robosuite -> RoboCasa` 迁移做准备。
- 区分“演示代码路径”和“正式运行路径”，避免 demo cycle 干扰主流程。

完成标准：

- 主流程不再依赖大量脚本级全局常量。
- 抓取、转运、推障、日志写入都能以函数或类接口独立调用。
- 替换环境层时，不需要改动 `vision/`、`ipc/`、`fsm/`、`logs/` 的协议。

### 2. 感知完整化

目标：把“能出结果”提升到“结果可置信、可量化、可迁移”。

建议：

- 完成真实 `T_world_cam` 标定。
- 记录感知延迟、置信度分布、空检测率、深度无效率。
- 增强多目标竞争和目标遮挡时的选择逻辑。
- 为视觉抓取点增加更强的语义约束，例如长轴估计、边缘抓取候选、把手优先策略。

完成标准：

- 视觉输出世界坐标误差有可报告的数值。
- `detected_objects` 低置信度和无效深度路径都有稳定降级行为。
- 抓取点生成逻辑不再强依赖仿真真值修正。

### 3. 执行完整化

目标：把当前启发式控制从“能演示”推进到“可复现、可调优、可诊断”。

建议：

- 系统化调优 `MOVE_GAIN`、`WAYPOINT_TOLERANCE`、分阶段 waypoint 和抓取前对齐阈值。
- 增加每个执行阶段的显式成功判据，而不只是 waypoint 是否接近。
- 增强失败恢复策略，例如双臂重新对齐、局部微调、单臂补偿。
- 将抓取、搬运、放置、推障的失败原因进一步细分。

完成标准：

- 日志能明确指出失败发生在 `transit / align / final_descent / close / lift / transfer / release` 哪一段。
- 多次重复运行时，成功率和误差分布有统计意义。
- 执行层调优能够基于日志闭环，而不是只依赖主观观察。

### 4. FSM 完整化

目标：让状态机从“基本通路”变成“严格实现任务协议的调度器”。

建议：

- 严格对齐 AGENTS.md 中的状态转换表和重试预算。
- 明确超时、重试耗尽、碰撞、无效感知的状态出口。
- 将 `FAILED -> IDLE`、`FAILED -> RESET`、`EMERGENCY` 的后处理路径写清楚。
- 把感知错误、执行偏移、物理滑落之外的内部子原因保留到日志中。

完成标准：

- FSM 的每个状态都有进入条件、退出条件、失败条件。
- 运行结束时的最终状态与 run log 中的失败分类保持一致。

### 5. 工程与协作完整化

目标：让项目对组员和评审都更可复现、更可维护。

建议：

- 修复依赖冲突，尤其是 `vision/detector.py` 对 `transformers>=4.38.0` 的要求，与 `requirements.txt` / `environment.yml` 当前版本保持一致。
- 补充最小测试集和运行前自检。
- 清理 README、安装步骤、模型缓存说明、常见报错说明。
- 固化标准实验命令、日志输出位置、配置版本命名规范。

完成标准：

- 新成员按文档可以独立启动一次完整实验。
- 环境问题、依赖问题、模型缓存问题都有明确排查路径。

### 6. 评估与报告完整化

目标：让项目最终不仅有代码，还有稳定的实验结果和分析材料。

建议：

- 固定 benchmark 场景、随机种子、对象数量和场景编号。
- 统一统计成功率、`N1/N2/N3`、平均时长、阶段失败分布。
- 扩展日志，记录 perception latency、dual-arm alignment error、gripper width、lift delta、transfer slip。
- 形成表格和图表输出模板，供最终报告直接使用。

完成标准：

- 每轮改动都可以通过相同基线实验进行前后对比。
- 最终报告中的结论可由 logs 自动回溯。

## 五、RL 接入设计方案

### 1. 当前阶段结论

当前阶段**不建议优先做重的 RL 训练或端到端视觉策略学习**。更合适的顺序是先把参数级调优、执行稳定性、相机标定和日志基线做扎实，再引入 RL。

原因：

- 当前主要瓶颈是执行稳定性和标定，而不是完全不会动。
- 双臂协作是高维控制问题，训练成本和调参成本都高。
- 如果观测、标定、失败标签、成功定义还不稳定，RL 学到的往往只是当前噪声补偿，难以迁移。

### 2. RL 的合理定位

RL 在本项目中的合理角色不是替代整套系统，而是替代一个**局部、短时域、难手调的控制子问题**。

建议保留：

- `vision/` 继续负责生成 `detected_objects`
- `ipc/` 保持 `perception_queue`
- `fsm/` 继续负责高层状态切换
- `logs/` 继续负责统一 run log

建议优先尝试用 RL 替代：

- 双臂抓取前最后一段对齐和闭合控制
- 抓取失败后的局部恢复动作
- 需要细致接触补偿的近场搬运微调

不建议第一阶段就做：

- 端到端图像输入抓取
- 完全替代 FSM 的高层任务策略
- 同时训练感知和控制

### 3. 推荐的第一阶段 RL 任务

最合适的切入点是：

- 从 `pre_grasp / align / final_descent / close_grippers` 这段短 horizon 子任务开始
- 保持放置、推障、日志、失败分类仍由现有逻辑负责

原因：

- 当前手写控制在这段最难调；
- 成功判据相对清晰；
- 仿真中可快速重复采样；
- 不会破坏现有整体架构。

### 4. 推荐的 RL 观测、动作、奖励

推荐先使用状态量而不是图像输入。

观测建议：

- `robot0_eef_pos`、`robot1_eef_pos`
- `robot0_eef_quat`、`robot1_eef_quat` 或 yaw 简化量
- 左右抓取目标点
- 当前目标物体位置或把手位置
- gripper width
- 最近一步的 alignment error

动作建议：

- 两臂末端位姿增量的缩放版本
- 两侧 gripper 开合命令
- 必要时只学习平移和夹爪，先固定姿态控制

奖励建议：

- 接近目标点奖励
- 双臂对齐误差减小奖励
- 成功闭合且抬升的奖励
- 发生碰撞、掉落、过大误差的惩罚
- 时间步惩罚，避免无效停留

### 5. RL 训练前置条件

只有当下面条件基本满足后，再进入 RL 开发窗口：

- `detected_objects` 输出稳定，低置信度和无效数据能正确降级
- `perception_queue -> FSM -> arm execution` 闭环能连续运行
- `T_world_cam` 已完成真实标定或至少有可量化误差
- 已积累足够多的成功 / 失败轨迹用于分析
- 现有启发式控制已经接近调参瓶颈

### 6. RL 成功判据

RL 不是“训出能动就算成功”，而应以对比基线判定：

- 在固定场景和固定随机种子上，抓取成功率高于当前启发式基线
- `N2/N3` 中至少一类失败显著下降
- 平均执行时长没有明显恶化
- 对少量新物体或新姿态具有比启发式更好的泛化

### 7. RL 的主要风险

- 训练成本高，时间预算可能不适合课程项目主线
- 双臂任务 reward 设计复杂，容易出现投机策略
- RL policy 可能在训练场景有效，但对新场景退化明显
- 如果直接做图像输入，训练与调试成本会明显上升

### 8. RL 路线建议

建议按下面顺序推进：

1. 先补齐标定、日志、测试和基线实验。
2. 固化当前启发式控制器的 benchmark 结果。
3. 抽出一个短 horizon 抓取子任务做单独环境包装。
4. 先用状态输入训练 RL policy。
5. 与现有启发式控制做 A/B 对比。
6. 只有在确实优于启发式后，再考虑扩大 RL 作用范围。

## 六、下一步最优先的改进方向

### 优先级 1：修复环境与工程基线

近期必须先解决：

- `transformers` 版本要求与依赖文件不一致的问题；
- README / 环境搭建 / 模型缓存说明不足的问题；
- 缺少最小测试集的问题。

这是后续所有实验、调参、训练工作的基础。

### 优先级 2：提高双臂执行收敛能力

这是当前最重要的算法问题。

下一步应优先处理：

- 双臂 waypoint 到位能力不足的问题；
- 某一侧机械臂经常未真正到达目标点的问题；
- 双臂闭合前缺少更稳定对齐过程的问题；
- 搬运与放置阶段对接触和姿态变化适应不足的问题。

建议方向：

- 继续增强分阶段 waypoint 设计；
- 在闭合夹爪前输出每只手的最终误差；
- 重新调节 `MOVE_GAIN`、双臂步进上限与容差；
- 增加抓取后抬升和转运的阶段性成功判据。

### 优先级 3：完成真实外参标定

应尽快完成：

- `T_world_cam` 标定；
- 去掉对真实物体中心的仿真修正依赖；
- 用仿真真值或标定板验证视觉世界坐标误差。

### 优先级 4：完善任务编排与模块边界

应继续完成：

- 将更多逻辑从 `main.py` 中拆出；
- 固化 perception / planning / execution / evaluation 的边界；
- 为 RoboCasa 迁移准备更稳定的 wrapper 层。

### 优先级 5：准备 RL 接入条件

当以上四项稳定后，再开始：

- 采集标准化轨迹数据；
- 固定 benchmark 场景；
- 抽出近场抓取子任务环境；
- 训练并比较第一个局部 RL policy。

## 七、后续待完成任务清单

### 1. 近期任务

1. 修复 `transformers` 版本要求与依赖文件不一致的问题。
2. 为 `detected_objects`、FSM 状态转移、run log、成功 / 失败判定补最小测试。
3. 提高双臂 waypoint 到位稳定性。
4. 为双臂抓取增加更明确的分阶段接近与对齐路径。
5. 在双臂闭合前记录每只手与目标点的最终误差。
6. 根据新日志内容重新调节：
   - `dual_grasp_lateral_offset`
   - `dual_grasp_height_offset`
   - `sim_xy_correction_alpha`
   - `MOVE_GAIN`
   - waypoint 容差与分段参数
7. 重新进行多轮实验，对比修改前后的抓取成功率。

### 2. 中期任务

1. 完成 `T_world_cam` 的真实标定。
2. 逐步减少并最终去掉仿真侧的抓取点修正逻辑。
3. 继续完善基于真实视觉的障碍物清理策略。
4. 将更多逻辑从 `main.py` 中拆出，形成更清晰的任务调度结构。
5. 固化 benchmark 场景、随机种子、对象数量与日志字段。
6. 扩展实验统计，包括：
   - perception latency
   - dual-arm alignment error
   - transfer slip
   - place accuracy
7. 整理 RL 前置数据，形成可训练的近场抓取子任务数据集或回放集。

### 3. 后期任务

1. 以环境封装层为边界，从 `robosuite` 迁移到 RoboCasa。
2. 在迁移过程中保持 `vision/`、`ipc/`、`fsm/`、`logs/` 接口稳定。
3. 启动局部 RL 训练实验，并与启发式控制器做 A/B 对比。
4. 若 RL 有明显收益，再考虑逐步扩大其作用范围。
5. 扩展实验记录能力，用于最终报告：
   - 抓取目标诊断数据
   - 双臂收敛误差
   - 感知延迟
   - 不同随机种子的成功率
   - RL 与启发式对比结果

## 八、验收标准

### 1. 基线工程验收

- 依赖环境可按文档复现；
- 主流程可稳定启动；
- run log 可正确写出；
- 最小测试集可运行并通过。

### 2. 算法与控制验收

- 双臂抓取成功率相较当前基线有稳定提升；
- `N1/N2/N3` 至少一项有明确下降；
- 执行层日志足以定位主要失败阶段；
- 视觉世界坐标误差有量化报告。

### 3. RL 验收

- RL policy 在固定 benchmark 上优于启发式基线；
- RL 失败类型可解释，且不会破坏现有 FSM 与日志协议；
- RL 只作为局部能力增强，而不是引入无法维护的新黑箱。

## 九、2026-04-26 Execution Layer Update

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
- The current arm motion strategy uses segmented, stepwise waypoint moves for stability during dual-arm carry and release. This can make the arms feel more hesitant or "stuck" than a larger-step controller, so a future follow-up may restore larger move steps once contact stability is confirmed.

## 十、2026-04-28 阶段结论

### 0. 本轮新增工程基线进展（协议冻结）

- 已新增 `tests/test_protocol_contracts.py`，为 `vision/`、`ipc/`、`fsm/`、`logs/` 四个稳定接口层补齐最小协议冻结测试。
- 当前已覆盖的契约包括：
  - `detected_objects` schema、低置信度降级、`time.monotonic()` 时间戳、障碍物上限截断；
  - `perception_queue` 命名与 `queue.get(timeout=0.2)` stale-read 语义；
  - FSM 状态枚举、`n1/n2/n3` 失败计数与重试耗尽路径；
  - run log 顶层字段、failure mode / failure_modes_triggered 语义。
- 已运行：`python -m unittest tests.test_protocol_contracts`
- 当前结果：13 个测试全部通过。
- 这意味着后续进入环境层迁移或 RL 基线接入前，`vision/`、`ipc/`、`fsm/`、`logs/` 已经具备可回归验证的协议护栏。

### 0.1 本轮新增环境与 RL 契约进展

- `arm/env_wrapper.py` 已从单一 robosuite 包装器扩展为稳定 contract：新增 `BaseEnvWrapper`、保留 `RobosuiteEnvWrapper`，并加入 `create_env_wrapper()` backend selector。
- 当前已明确 backend 行为：
  - `backend="robosuite"` 走现有实现；
  - `backend="robocasa"` 当前会显式抛出 `NotImplementedError`，而不是静默 fallback；
  - 未知 backend 会显式抛出 `ValueError`。
- 已新增 `tests/test_env_wrapper_contract.py`，验证 wrapper contract 类型关系、unsupported backend 失败路径、RoboCasa 未实现路径。
- 为 RoboCasa + RL 主线的下一步接入，已新增最小 RL 契约与 dry-run harness：
  - `configs/rl/dual_arm_robocasa.yaml`
  - `rl/contracts.py`
  - `rl/harness.py`
  - `scripts/train_rl.py`
  - `scripts/eval_rl.py`
  - `tests/test_rl_contracts.py`
- 当前 RL 相关实现仍是 **contract/harness 层**，不是实际 RoboCasa policy 训练：
  - 已固定 milestone-1 config schema、reward/observation/action/horizon 字段；
  - 已支持 dry-run checkpoint 与 metrics artifact 生成；
  - 已支持缺失 checkpoint / 缺失配置字段的明确失败路径。
- 已运行：`python -m unittest tests.test_protocol_contracts tests.test_env_wrapper_contract tests.test_rl_contracts`
- 当前结果：22 个测试全部通过。
- 当前阻塞项也已明确：仓库仍未接入真实 RoboCasa backend adapter，也未落地实际 RL trainer/evaluator，因此后续任务需要补齐外部依赖与运行时集成，不能把 dry-run harness 误当成真实训练完成。

### 0.2 本轮新增 RoboCasa / RL 依赖基线进展

- 已新增 **独立于现有 robosuite/mink 基线** 的 RoboCasa / RL 迁移依赖文件：
  - `requirements-robocasa-rl.txt`
  - `environment-robocasa-rl.yml`
- 这两份文件没有覆盖现有 `requirements.txt` 与 `environment.yml`，原因是当前项目主线仍然依赖 Python 3.10 + robosuite/mink，而 RoboCasa 官方安装与本地真实安装都已经把依赖解析推到了另一组版本：`numpy 2.2.5`、`torch 2.7.1`、`torchvision 0.22.1`、`gymnasium 0.29.1`。
- 由于 `mink 0.0.5` 明确要求 `numpy<2.0.0`，当前仓库应把 RoboCasa/RL 环境视为**单独环境**，不能再默认“robosuite + mink + RoboCasa”共用一个可同时满足的 baseline。
- 已新增最小依赖 smoke / setup 辅助脚本：
  - `scripts/check_robocasa_rl_deps.py`
  - `scripts/print_robocasa_rl_setup.py`
- 当前依赖基线选择结论：
  - RoboCasa/RL 环境与 robosuite/mink 基线分离；
  - RL trainer 最小推荐基线选用 `torch + gymnasium + stable-baselines3`；
  - 当前 repo 自定义的 `rl/contracts.py` / `rl/harness.py` 负责本地 contract 与 dry-run，而不是替代真实 trainer。
- 已根据真实安装结果修正 RoboCasa / RL 专用依赖基线：`numpy=2.2.5`、`torch=2.7.1`、`torchvision=0.22.1`、`gymnasium=0.29.1`；这组版本与 mink 的约束冲突，因此不能并入旧的 robosuite/mink 基线。
- 已新增 `tests/test_dependency_baseline.py`，验证：
  - 新依赖文件存在且包含 RoboCasa / RL 基线字段；
  - setup 辅助脚本可执行；
  - dependency smoke script 能输出结构化 JSON，并明确标注 `separate_robocasa_rl_env` 与 `mink_numpy_conflict`。
- 已运行：
  - `python -m unittest tests.test_protocol_contracts tests.test_env_wrapper_contract tests.test_rl_contracts tests.test_dependency_baseline`
  - `python scripts/validate_task_roadmap.py`
- 当前结果：26 个测试全部通过，roadmap 校验通过。
- 当前 blocker / 风险进一步收敛为两类：
  1. 还未实际安装 `robocasa` 包及其厨房资产，因此 `robocasa_ready` 仍不能假设为真；
  2. RoboCasa/RL 环境与 robosuite/mink baseline 已确认不能统一，后续任何联调都必须先选定目标环境，避免把 numpy 2.x 直接塞回 mink 环境。

### 0.3 本轮新增单臂 RoboCasa RL 实现进展

- 已新增最小单臂训练配置：`configs/rl/single_arm_robocasa_ppo.yaml`。
- 已把 `arm/env_wrapper.py` 扩展为最小 `RobocasaEnvWrapper`，当前支持：
  - `backend="robocasa"` 进入独立 wrapper；
  - `reset / step / render / get_flat_observation` 的最小 RL 路径；
  - 通过 Gymnasium `gym.make("robocasa/<Task>")` 创建环境。
- 已把 `rl/contracts.py` 扩展到支持 `render` 段。
- 已把 `rl/harness.py` 从纯 dry-run 扩展到：
  - `run_train_smoke(..., dry_run=False)` 走真实 SB3 PPO 训练；
  - `run_eval_smoke(..., dry_run=False)` 走真实 checkpoint load + rollout；
  - `get_latest_checkpoint(...)` 支持最新 checkpoint 解析。
- 已把 CLI 扩展为：
  - `scripts/train_rl.py --timesteps ...`
  - `scripts/eval_rl.py --latest`
- 已新增/更新测试以覆盖：
  - 单臂 config contract；
  - RoboCasa adapter seam；
  - 旧双臂 config 兼容未被破坏。
- 已运行：`python -m unittest tests.test_rl_contracts tests.test_env_wrapper_contract`
- 当前结果：10 个测试全部通过。

### 0.4 本轮真实训练尝试结果（重要）

- 已在独立环境 `robotic-robocasa-rl` 中执行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 200`
- 真实运行顺序已经推进到：
  1. CLI 进入真实 `run_train_smoke(..., dry_run=False)`；
  2. 读取单臂 config；
  3. 构造 RoboCasa Gym env；
  4. 进入 RoboCasa `create_env(...) -> robosuite.make(...)`；
  5. 在底层环境构造阶段失败。
- 已修复的本地代码问题包括：
  - 单臂 config 缺少 `env.task_name`；
  - `split` 参数未透传；
  - `render_mode` 不应在训练阶段透传给 RoboCasa Gym wrapper。
- 当前**真实 blocker** 已确认不是本地脚手架缺失，而是外部版本兼容：
  - RoboCasa 环境构造最终报错：`ManipulationEnv.__init__() got an unexpected keyword argument 'load_model_on_init'`
  - 这说明当前安装组合下，`robocasa 1.0.0` 与现有 `robosuite 1.5.2` 的环境构造参数接口不兼容。
- 结论：
  - 本地仓库已经具备“单臂训练 + checkpoint + rollout”最小代码路径；
  - 今天没有打通真实训练，不是因为 repo 里仍缺 trainer/runner，而是因为 **RoboCasa ↔ robosuite 版本组合** 阻塞了环境初始化。
  - 下一步优先级应转向：确认 RoboCasa 官方要求的 robosuite 版本，必要时在独立环境中调整 `robosuite` 版本组合后再重试训练。

### 0.4.1 RoboCasa 重试进展（资产与 adapter 修复后）

- 在下载完 kitchen 资产并将独立环境中的 `robosuite` 切到源码版后，当前已重新尝试：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 200`
- 这轮重试已经连续解决了几类前置阻塞：
  - kitchen 资产缺失导致的 `model.xml` 找不到问题已消失；
  - `load_model_on_init` 兼容问题已通过源码版 `robosuite` 解决；
  - `split` 参数透传与 `render_mode` 误透传问题已修复；
  - RoboCasa Gym action space 不是 Box 而是 `gymnasium.spaces.Dict` 的问题已识别并开始适配；
  - observation 中包含字符串任务描述，导致 flatten 失败的问题已修复。
- 当前最新的真实 blocker 已推进到：
  - **RoboCasa Dict action 的编码/解码映射仍不正确**；
  - 具体报错出现在 `arm/env_wrapper.py -> RobocasaEnvWrapper.format_action()`：
    - `ValueError: can only convert an array of size 1 to a Python scalar`
  - 这说明 RoboCasa Gym wrapper 的 Dict action 不仅需要 flatten/unflatten，还需要与其 `key_converter` 的真实期望顺序和标量/向量类型严格对齐。
- 结论：
  - RoboCasa 路线已经不再卡在“外部依赖是否存在”这一级；
  - 当前 blocker 已经转化为 **本地 runner 对 RoboCasa Dict action space 的精确适配问题**；
  - 也就是说，把当前已经跑通的 robosuite 单臂训练/可视化 runner 迁回 RoboCasa，剩下的关键工作已缩小到 action mapping，而不是整条链路重写。

### 0.4.2 RoboCasa 训练恢复与评估卡住现状

- 在继续修正 RoboCasa adapter 后，当前已进一步解决：
  - RoboCasa Gym action space 实际为 `gymnasium.spaces.Dict`，不是单一 Box；
  - `single_arm_robocasa_ppo.yaml` 的 action 维度已从错误的 `4` 修正为按 RoboCasa wrapper 推断得到的 `12`；
  - Dict action 中的单元素字段已不再按错误的标量路径处理；
  - observation flatten 现在会跳过字符串任务描述等非数值字段。
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 200`
- 当前结果：
  - RoboCasa 单臂训练已成功落盘 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robocasa_reach_grasp-ppo-200.zip`
- 之后继续运行：
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --render`
- 当前评估现象：
  - 不再立即因 env 构造、资产缺失或 action mapping 报错退出；
  - 运行会进入 `reset()` / `step()`，但在 `--render` 路径上长时间停留，当前表现为“看起来卡住”，需要人工中断；
  - 日志里可见 Gymnasium 的 observation-space warning，但这不是当前最主要的阻塞点。
- 当前状态应理解为：
  - **RoboCasa 单臂训练主链路已经开始工作并能落 checkpoint**；
  - **RoboCasa 评估/可视化链路尚未稳定，当前 `--render` 路径会卡住**；
  - 下一步应优先把评估 runner 改成“无渲染先完成 / 有渲染时带 step 级进度与 hard max-step guard”，先判断是 render 卡住，还是 rollout 本身过慢。

### 0.4.3 RoboCasa 训练/评估进度可视化与 CPU 现状

- 当前已进一步增强训练与评估可见性：
  - 训练开始时会打印 `[train] backend=... task=... total_timesteps=... n_steps=...`；
  - 已新增 heartbeat callback，默认每 20 个 timestep 打印一次 `[train] timesteps=...`；
  - PPO `n_steps` 已调小到 `64`，便于更快看到第一轮统计输出。
- 已重新执行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000`
- 真实现象已经确认：
  - RoboCasa 训练**不是完全卡死**；
  - 它可以进入 `Using cpu device`、`Monitor`、`DummyVecEnv`，并持续打印 heartbeat 和第一轮 SB3 统计；
  - 但推进速度明显很慢，第一轮 `64` step 用时较长。
- 当前环境的 PyTorch 状态也已确认：
  - `torch_version = 2.7.1+cpu`
  - `torch.cuda.is_available() = False`
  - 当前 `robotic-robocasa-rl` 环境没有可用 CUDA，因此 RoboCasa 训练只能走 CPU。
- 在评估侧，当前已确认：
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest`（**不带 render**）能够结束并写出 metrics；
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --render` 仍然容易长时间停留，因此当前 render 路径还不应视为稳定。
- 当前状态应理解为：
  - RoboCasa 路线已经从“训练起不来”推进到“训练能开始、无渲染评估能完成，但 CPU 很慢、render 仍不稳定”；
  - 如果后续要进一步提高训练效率，优先级已经转向：为 `robotic-robocasa-rl` 环境切换 CUDA 版 PyTorch，而不是继续单纯等待 CPU 训练。

### 0.4.4 RoboCasa reward shaping 与 5000-step 训练结果

- 在确认 RoboCasa 高层 Gym observation 里没有直接的目标物体位姿后，当前已下探到 RoboCasa 底层 raw observation，并确认可用字段包括：
  - `obj_pos`
  - `obj_quat`
  - `obj_to_robot0_eef_pos`
  - `robot0_gripper_qpos`
  - 以及其他 `robot0_*` proprioception 字段
- 基于这些底层字段，当前已把 RoboCasa 路线的 reward shaping 真实接入 `RLVectorEnvAdapter`，不再只依赖环境默认 reward。当前启用的最小 shaping 包括：
  - `approach_target`
  - `successful_close`
  - `successful_lift`
  - `timeout_penalty`
- 同时，`single_arm_robocasa_ppo.yaml` 已升级为更像样的 baseline：
  - `action.dimensions` 修正为 `12`，与 RoboCasa Dict action space 对齐；
  - `approach_target: 2.0`
  - `successful_close: 4.0`
  - `successful_lift: 6.0`
  - `timeout_penalty: -0.05`
  - `train.total_timesteps: 5000`
  - `train.n_steps: 64`
  - `train.progress_print_freq: 20`
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest`
- 当前结果：
  - RoboCasa 5000-step 训练已成功落盘 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robocasa_reach_grasp-ppo-5000.zip`
  - RoboCasa 无渲染评估已成功落盘 metrics：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-metrics.json`
  - 训练日志已能持续打印 heartbeat 与 SB3 统计表，确认训练在真实推进，而不是假死。
- 这意味着：
  - RoboCasa 单臂路线已经实现了 **训练 → checkpoint → 无渲染 eval** 的完整闭环；
  - 当前剩余未完全确认的主要项已经缩小为：`--render` 可视化路径的稳定性，以及策略质量本身是否足够好看。

### 0.4.5 RoboCasa 训练后模型视频可视化结果

- 进一步核对 RoboCasa 官方 Gym wrapper 后，当前已确认：`render()` 返回的是缓存图像帧 `render_cache`，并不等同于稳定的 onscreen viewer。
- 基于这个事实，当前已把 RoboCasa 可视化主路径从“优先尝试实时窗口显示”切换为更稳的 **offscreen frame capture + mp4 输出**。
- 当前已在本地实现：
  - `scripts/eval_rl.py` 新增 `--save-video`
  - `scripts/eval_rl.py` 新增 `--video-path`
  - `rl/harness.py` 在评估 loop 中逐步抓取 `env.render()` 返回的帧，并通过 `imageio` 写出视频。
- 已真实运行：
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --save-video`
- 当前结果：
  - RoboCasa rollout 视频已成功生成：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-rollout.mp4`
  - 对应 metrics 也已同步生成：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-metrics.json`
- 当前状态应理解为：
  - 你要求的“**真正看到 RoboCasa 里训练后模型的可视化**”现在已经有了**可交付的视频结果**；
  - 当前更推荐的视频可视化路径是：**训练 → checkpoint → 无 viewer rollout → mp4 输出**，而不是继续依赖不稳定的 `--render` 实时窗口模式。

### 0.4.6 RoboCasa 抓取导向收敛强化结果

- 在进一步明确当前阶段目标是“先稳定夹起物体”，而不是完整放进 cabinet 后，当前已把 RoboCasa 单臂路线进一步压缩成更抓取导向的训练目标。
- 当前新增的关键策略不是单纯增加训练步数，而是：
  - 在 `RLVectorEnvAdapter.step()` 中加入**抓取成功提前终止**逻辑：当距离足够近、夹爪足够闭合、物体相对初始高度已经抬升超过阈值时，提前把当前 episode 视为成功终止；
  - 加入“抬起后掉落”的额外惩罚，使策略更倾向于稳定保持抓取，而不是偶然带起后又掉落。
- 同时，`single_arm_robocasa_ppo.yaml` 也已进一步调优：
  - `successful_close: 5.0`
  - `successful_lift: 8.0`
  - `timeout_penalty: -0.05`
  - `termination.max_steps: 120`
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest`
- 当前结果：
  - 抓取导向 RoboCasa checkpoint 已更新：`.sisyphus/artifacts/rl/checkpoints/single_arm_robocasa_reach_grasp-ppo-5000.zip`
  - 无渲染评估 metrics 已更新：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-metrics.json`
  - 当前 eval episode 能在 `120` step 范围内结束，而不再只是之前那种更长的固定拖尾。
- 当前状态应理解为：
  - RoboCasa 路线现在已经不只是“能训练、能评估、能出视频”，而是更明确地朝“**先稳定夹起物体**”的阶段性目标收敛；
  - 它仍然不是最终的 cabinet 放置策略，但已经更贴近你当前想要的“稳定夹取物体”的方向。

### 0.4.7 RoboCasa grasp-only 收缩版 baseline 结果

- 在继续分析“视频里主要只看到机械臂环绕自转、看不清目标物体”的问题后，当前已确认两个关键症结：
  1. RoboCasa 训练时原始动作空间过大，策略同时学习 gripper、末端位移、末端旋转、底座运动与 control mode，容易先学出无效的绕圈动作；
  2. RoboCasa 高层 observation 同时包含图像和文本，直接 flatten 后不适合作为 `MlpPolicy` 的干净抓取输入。
- 因此，当前已把 RoboCasa 路线进一步收缩成 **grasp-only baseline**：
  - 动作上，训练只保留：
    - `action.gripper_close`（1 维）
    - `action.end_effector_position`（3 维）
  - 其余动作块：
    - `action.end_effector_rotation`
    - `action.base_motion`
    - `action.control_mode`
    在训练时固定为零，不再参与策略学习。
- 在观测上，当前已明确改为：
  - **只保留 `state.*` 数值字段** 输入给 `MlpPolicy`；
  - 不再把三路 `video.*` 图像或文本任务描述 flatten 进训练输入；
  - reward shaping 仍可从底层 raw obs 读取 `obj_pos`、`obj_to_robot0_eef_pos`、`robot0_gripper_qpos` 等信号。
- 在可视化上，当前也已把 RoboCasa 视频输出的默认帧源切换为更利于看抓取的：
  - `video.robot0_eye_in_hand`
  而不是原来的左侧固定视角。
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --save-video`
- 当前结果：
  - grasp-only 训练已成功更新 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robocasa_reach_grasp-ppo-5000.zip`
  - grasp-only 视频已成功更新：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-rollout.mp4`
  - 对应 metrics 也已更新：`.sisyphus/artifacts/rl/metrics/single_arm_robocasa_reach_grasp-eval-metrics.json`
- 当前状态应理解为：
  - RoboCasa 路线现在已经不只是“能输出视频”，而是已经收缩成一个更干净、更接近“夹取物体”的单臂 baseline；
  - 如果后续还想进一步提升效果，下一步最值的方向将是继续扩大训练步数，或者再把成功判据和接触级 reward 做得更细。

### 0.5 本轮 robosuite 单臂 smoke 训练 / 可视化验证结果

- 为了在 RoboCasa kitchen 资产下载和版本兼容问题之外，先验证“训练后模型可视化”主链路，当前已新增一条 **robosuite 单臂 smoke path**：
  - `configs/rl/single_arm_robosuite_ppo.yaml`
- 已对本地实现做的最小补强包括：
  - `RobosuiteEnvWrapper` 新增 `get_flat_observation()`，用于 privileged-state RL baseline；
  - `RobosuiteEnvWrapper` 现在兼容单臂 observation，不再强制读取 `robot1_*` 字段；
  - `get_observation()` 允许无 camera obs 的路径；
  - `RLVectorEnvAdapter` 现在显式继承 `gymnasium.Env`，可被 SB3 正常包装；
  - RL contract 允许 `backend` 为 `robosuite` 或 `robocasa`。
- 已运行：
  - `python -m unittest tests.test_rl_contracts tests.test_env_wrapper_contract`
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 100`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest`
- 当前结果：
  - 11 个测试全部通过；
  - 真实训练成功落盘 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robosuite_lift_smoke-ppo-100.zip`
  - 真实评估成功落盘 metrics：`.sisyphus/artifacts/rl/metrics/single_arm_robosuite_lift_smoke-eval-metrics.json`
- 这意味着今天已经在 **robosuite 单臂环境** 中打通了：
  - 训练
  - checkpoint 保存
  - checkpoint 加载
  - rollout 执行
  - render 路径调用
- 当前状态应理解为：
  - **训练后模型可视化主链路已在 robosuite smoke path 上跑通**；
  - **RoboCasa 路线仍卡在外部环境/资产层**，而不是本地训练脚手架层。

### 0.6 本轮 robosuite 单臂 baseline 强化结果

- 在 smoke path 跑通后，当前已将 `configs/rl/single_arm_robosuite_ppo.yaml` 从最小 smoke 配置提升为更像样的 baseline：
  - `task_name` 从 `single_arm_robosuite_lift_smoke` 更新为 `single_arm_robosuite_lift_baseline`；
  - `train.total_timesteps` 从 `200` 提高到 `5000`；
  - `termination.max_steps` 从 `100` 提高到 `150`；
  - `eval.episodes` 从 `1` 提高到 `3`；
  - 训练阶段关闭 renderer，评估阶段通过 `--render` 单独打开，避免训练时的渲染开销拖慢学习。
- 为支撑这个 baseline，当前已进一步修正本地实现：
  - `RobosuiteEnvWrapper` 兼容单臂路径，不再强制要求 `robot1_*` 字段；
  - `get_observation()` 允许 `use_camera_obs: false` 时不读取 `frontview_image` / `frontview_depth`；
  - `RLVectorEnvAdapter` 现在显式继承 `gymnasium.Env`；
  - 训练与评估的 renderer 配置已拆分，评估时可单独启用可视化。
- 已运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest --render`
- 当前结果：
  - 更长训练成功落盘 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robosuite_lift_baseline-ppo-5000.zip`
  - 评估成功落盘 metrics：`.sisyphus/artifacts/rl/metrics/single_arm_robosuite_lift_baseline-eval-metrics.json`
- 这意味着：
  - 当前 repo 已经不只是“训练/加载/可视化 smoke 测试”，而是有了一个更接近真实学习的 **robosuite 单臂 cube-lift PPO baseline**；
  - 视觉效果如果仍然一般，主要原因将不再是链路没打通，而是 reward / timesteps / task定义本身仍然很弱，需要继续训练与调参。

### 0.7 本轮 reward shaping 接入结果

- 经过代码核对，之前 `configs/rl/single_arm_robosuite_ppo.yaml` 中的 reward 系数虽然写在配置里，但**并未真正接入训练逻辑**；PPO 实际学习的是 robosuite `Lift` 环境原生 reward。
- 当前已把 reward shaping 真正接入 `rl/harness.py` 的 `RLVectorEnvAdapter.step()`，第一版接入了三类最小有效项：
  - `approach_target`
  - `successful_lift`
  - `timeout_penalty`
- 当前 reward 逻辑要点：
  - 末端到 cube 距离越小，额外给 `approach_target` 正奖励；
  - cube 高度相对前一步上升时，额外给 `successful_lift` 奖励；
  - 每个 step 施加轻微 `timeout_penalty`，减少无效停留。
- 在 reward shaping 接入后，已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest --render`
- 当前结果：
  - 训练成功重新生成 baseline checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robosuite_lift_baseline-ppo-5000.zip`
  - 评估成功重新生成 metrics：`.sisyphus/artifacts/rl/metrics/single_arm_robosuite_lift_baseline-eval-metrics.json`
- 当前状态应理解为：
  - 现在的 baseline 已经不再只是依赖 robosuite 原生 reward；
  - 当前 reward shaping 仍然很轻量，尚未包含更复杂的闭合成功、抓取稳定性、碰撞或姿态约束项，因此后续仍有明显调优空间。

### 0.8 本轮从“抬起”向“夹取”导向的 baseline 强化结果

- 在进一步分析当前视觉效果后，已确认之前的 baseline 更像是“把 cube 抬起来”，而不是“明确学会夹取 cube”。原因在于之前 reward shaping 主要鼓励接近与抬升，对夹爪闭合动作本身的约束不够强。
- 当前已进一步调整 `single_arm_robosuite_ppo.yaml` 与 `RLVectorEnvAdapter.step()`，把 baseline 从“lift-oriented”往“grasp-oriented”推进：
  - `approach_target` 从 `1.0` 提高到 `2.0`
  - `successful_close` 从 `2.0` 提高到 `4.0`
  - `successful_lift` 从 `3.0` 提高到 `6.0`
  - `timeout_penalty` 从 `-0.5` 调整到 `-0.05`，避免策略在还不会闭合时被过强的时间惩罚压制
  - `train.total_timesteps` 从 `5000` 提高到 `20000`
- 当前 reward shaping 已新增更明确的夹取导向项：
  - 末端到 cube 距离改善量奖励，而不只是静态距离奖励；
  - 当末端已接近 cube 且夹爪宽度缩小时，给 `successful_close` 奖励；
  - 当夹爪宽度低于阈值时，再追加一次闭合奖励；
  - 对象高度不仅按逐步增量奖励，还按相对 episode 起点的累计抬升奖励。
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 20000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest --render`
- 当前结果：
  - 新的 checkpoint 已生成：`.sisyphus/artifacts/rl/checkpoints/single_arm_robosuite_lift_baseline-ppo-20000.zip`
  - 新的 metrics 已生成：`.sisyphus/artifacts/rl/metrics/single_arm_robosuite_lift_baseline-eval-metrics.json`
- 当前状态应理解为：
  - 这个 baseline 现在已经比之前更明确地鼓励“靠近 → 闭合 → 抬升”的抓取动作序列；
  - 但它仍然是 **robosuite Lift 单臂 baseline**，不是完整的高质量抓取 benchmark，因此后续仍可继续从 episode 成功判据、抓取稳定性奖励、碰撞惩罚和更长训练预算上继续强化。

### 0.9 本轮更细粒度抓取导向 reward 结果

- 在继续提升单臂 baseline 时，已先检查当前 `robosuite Lift` 高层观测/信息是否足以支持更细的接触型 reward。结果确认：
  - 当前 obs 中有：`cube_pos`、`cube_quat`、`gripper_to_cube_pos`、`robot0_*`、`object-state`
  - 当前 `env.step()` 返回的 `info` 为空，不提供现成的接触类型标记
  - 因此，像“夹子内侧接触 / 外侧接触 / 非夹子接触 / 击飞物体”这种 reward，当前**不能直接从高层 API 精确判断**。
- 基于这个约束，当前已把 reward 进一步升级为“可观测近似版夹取奖励”，重点依赖：
  - `gripper_to_cube_pos`
  - `robot0_gripper_qpos`
  - `cube_pos`
- 当前新增/增强的抓取导向项包括：
  - 末端到物体距离改善量奖励（比静态距离更强调持续逼近）；
  - 当距离足够近且夹爪宽度缩小时，给予闭合奖励；
  - 当夹爪宽度低于阈值时，再追加一次闭合奖励；
  - 当夹爪已接近且物体发生位移时：
    - 若夹爪仍张开，则把这种位移近似视为“推远/击飞”并施加惩罚；
    - 若夹爪已较闭合，则把这种位移近似视为“抓住后带动”并给予奖励；
  - 保留相对初始高度和逐步高度增量的抬升奖励。
- 已重新运行：
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 20000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest --render`
- 当前结果：
  - 训练成功更新 checkpoint：`.sisyphus/artifacts/rl/checkpoints/single_arm_robosuite_lift_baseline-ppo-20000.zip`
  - 评估成功更新 metrics：`.sisyphus/artifacts/rl/metrics/single_arm_robosuite_lift_baseline-eval-metrics.json`
- 当前状态应理解为：
  - 这套 reward 已经比之前更贴近“夹取物体”而不是“只把物体抬起来”；
  - 但它仍然是基于高层观测的近似抓取 reward，而不是基于 MuJoCo 接触对的精确接触 reward；
  - 如果后续要实现你提出的“夹子内侧接触 / 外侧接触 / 非夹子接触 / 掉落惩罚”这些更细粒度目标，下一步应下探到 MuJoCo contact 级别信号。

### 1. 关于项目主线

当前项目主线不应转向“大规模模型训练”或“端到端 RL”。更合理的路线仍然是：

- 先把工程基线、标定、执行稳定性和日志评估做扎实；
- 再在局部子任务上引入 RL 进行能力增强；
- 最后视收益情况决定是否扩大 RL 覆盖范围。

### 2. 当前最值得做的事情

- 继续提升双臂连续轨迹执行的速度与观感，减少阶段边界停顿。
- 继续优化抓取接近段的闭合窗口和 Z 方向收敛，减少 approach 末端的保守停顿。
- 优先修复 `push` 动作策略，因为当前主失败已从 pick-place 主链转移到 post-grasp push test。
- 清理连续轨迹重构后的日志残留问题，避免 `dual_arm_execution_diagnostics` 中出现“任务成功但 stage 仍残留 failed_stage”的假失败记录。
- 尽快完成 `T_world_cam` 的真实标定，减少对仿真修正的依赖。
- 保持 `vision/`、`ipc/`、`fsm/`、`logs/` 接口稳定，不要为了训练提前改接口。
- 修复依赖和测试基线问题，为后续实验和可能的 RL 训练打底。

### 3. 适合先调的参数

- `dual_grasp_lateral_offset`
- `dual_grasp_height_offset`
- `sim_xy_correction_alpha`
- `MOVE_GAIN`
- waypoint 步长与容差
- 相机内参和外参

### 4. 什么时候可以开始 RL

如果后续出现以下情况，就可以把 RL 提上日程：

- 参数已经多轮调整，但收益明显下降；
- 失败主要来自新物体、新场景或新任务分布；
- 控制策略已经稳定，问题更多表现为泛化不足；
- 任务数据和日志已经足以支持训练与回放分析；
- 基线启发式控制器已形成稳定 benchmark，可用于公平比较。

### 5. 当前执行层补充进展

- 已开始将单臂与双臂 waypoint 执行从离散目标追踪改为平滑参考跟踪。
- 当前实现方向是保留高层阶段语义，但在每个阶段内部引入 eased reference 和 action 变化率限制。
- 双臂抓取前半段 `transit -> pre_grasp -> align -> final_descent -> grasp_pose` 已开始合并为单次连续 `approach trajectory`。
- 双臂 `transfer` / `place` / `release` 主路径已开始从多段循环推进重构为单次连续轨迹跟踪。
- 放置成功判定已收紧为联合条件：目标 XY 到位、物体高度回到桌面附近、姿态保持基本直立、夹爪已释放，避免“被带到附近但未松夹”或“被打翻但发生位移”误判成功。
- 放置失败后的重试逻辑已开始分流：已到位但未释放时优先 `retry release`，已释放但未稳定时优先等待复查，只有真正未到位时才重跑完整 pick-place。
- 清障阶段已经从 push 切换为 grasp-and-relocate：障碍物会被抓起并搬到侧边，不再依赖推障动作。
- 当前已开始把 grasp 与清障分成独立 episode 记账，运行日志会分别记录两段的来源、成功/失败和前后物体位置，便于后续单独分析两类回合。
- 当前单臂执行也已接入整段轨迹连续化，`execute_grasp` 和 `execute_single_grasp_transfer` 不再只做离散 waypoint 逐点追踪。
- 当前归位逻辑已补上初始 joint 角校验：`home_arms` 现在不仅要求末端回到初始位姿，还会检查各 joint 是否回到 reset 时记录的初始角度。
- 默认仿真任务已经从 `TwoArmLift` 切换到 `Lift`，目标主体改为库内的 `cube`，感知标签和对象优先级也已同步调整。
- 下一步需要继续用日志和可视化验证：运动是否更连续、主链成功率是否保持、阶段切换是否减少停顿、清障搬运是否稳定不掉落。

### 6. 2026-04-28 最新状态总结

- 根据 `logs/20260428_173529.json`，当前系统已经可以完成 perception -> grasp -> lift -> transfer -> place 主链，且最终失败不再来自 pick-place 主流程。
- 当前整轮 `Run success` 的主要风险已经从推障偏移切换为清障搬运的稳定性，说明项目瓶颈已从“能否推开障碍”转移到“能否稳定抓取并搬离障碍”。
- 放置逻辑相关的两个关键判定漏洞已被修复：
  - 物体被打翻但恰好位移到附近，不再算成功。
  - 物体被带到目标附近但仍夹在手里，不再算成功。
- 当前最需要继续收尾的是两项：
  - 让连续轨迹控制在视觉上更自然、更少停顿；
  - 清理日志诊断残留，使阶段 success / failed_stage 与真实任务结果一致。

### 7. 推荐下一步（当前 work session）

- owner group: `execution` + `infra`
- goal: 在已完成 robosuite 单臂 baseline 训练/可视化闭环后，二选一推进：
  1. 继续提高 robosuite 单臂 cube-grasp / cube-lift baseline 的训练质量，并视需要下探到 MuJoCo 接触级 reward；
  2. 在 kitchen 资产和版本兼容问题解决后，把同一条 runner 切回 RoboCasa 单臂环境。
- success criteria:
  - `environment-robocasa-rl.yml` 对应的独立环境可创建并通过 `scripts/check_robocasa_rl_deps.py`；
  - RoboCasa backend 可以被 `create_env_wrapper(backend="robocasa")` 真正实例化；
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000` 能稳定训练并落盘 checkpoint；
  - train/eval harness 不再只是 dry-run，而能连接真实 backend 和实际 policy/rollout；
  - RoboCasa eval 至少要先在无渲染模式下稳定完成（已完成），并优先支持视频输出（已完成）；
  - 如果继续扩大 RoboCasa 训练预算，应优先解决当前 CPU-only 训练速度问题；
  - 与当前 `vision/`、`ipc/`、`fsm/`、`logs/` 协议保持兼容；
  - 不再假设 `mink 0.0.5` 与 numpy 2.x 共存。
- suggested verification:
  - `python -m unittest tests.test_protocol_contracts tests.test_env_wrapper_contract tests.test_rl_contracts tests.test_dependency_baseline`
  - `python scripts/check_robocasa_rl_deps.py`
  - `python scripts/train_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --timesteps 20000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robosuite_ppo.yaml --latest --render`
  - `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 5000`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest`
  - `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --save-video`
  - 在实时 render 路径恢复前，不要把 `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --render` 作为通过条件
