# 项目任务进度板

> 维护约束：本文件只记录需要进入 git 仓库的源码、配置、文档、接口状态和下一步任务。不要记录 `.venv` 重建、解释器安装、模型缓存下载、个人机器路径、临时运行日志等本地运行时操作。

## 当前目标

当前默认任务主线是**视觉驱动的单臂移动目标 + 单臂清障**，同时保留双臂连续轨迹与放置验证相关能力，作为后续扩展与回归基线：

- `robot0`：根据视觉目标位置抓取并移动目标物体。
- `robot1`：根据视觉障碍物位置执行清障推移。
- `vision -> perception_queue -> FSM -> execution -> logs` 链路保持稳定。
- 双臂 `approach / transfer / place / release` 连续轨迹、放置判定、push 判定修正继续保留，作为执行层稳定化成果。
- 视觉模型不可用时可以保留 observation fallback 用于控制调试，但不能把 fallback 误写成视觉链路已完成。

## 最近完成

- `infra/docs`: `AGENTS.md` 已补充 `docs/task.md` 维护协议，要求每次运行前读取、阶段完成后更新、按分组维护进度，并限制 `docs/task.md` 只记录需要进入 git 的源码/配置/文档/接口状态和下一步任务。
- `vision/perception`: 新增 `scripts/cache_grounding_dino.py`，用于按 `configs/vision.yaml` 中的 `model_id` 缓存并校验 GroundingDINO 模型；脚本会验证 `local_files_only=True` 下 processor 和 model 可加载。
- `vision/perception`: `vision/detector.py` 增强本地模型缺失时的错误提示，提示运行缓存脚本，而不是暴露底层加载异常。
- `integration/merge`: 默认任务主线已切换为“单臂抓取 + 单臂清障”。`main.py` 默认 `GRASP_MODE = "single"`，并通过 `single_vision_grasp_offset` 控制视觉抓取点修正。
- `logic/fsm`: `fsm/perception_cycle.py` 在发布新视觉 payload 前会清空旧队列，避免清障后读取过期感知结果。
- `control/execution`: 保留并扩展执行层连续化成果：
  - 单臂与双臂 waypoint 执行已从离散目标追踪推进为平滑参考跟踪。
  - 双臂抓取前半段 `transit -> pre_grasp -> align -> final_descent -> grasp_pose` 已开始合并为单次连续 `approach trajectory`。
  - 双臂 `transfer / place / release` 主路径已开始从多段循环推进重构为单次连续轨迹跟踪。
- `control/validation`: 放置成功判定已收紧为联合条件：目标 XY 到位、物体高度回到桌面附近、姿态保持基本直立、夹爪已释放，避免“被带到附近但未松夹”或“被打翻但发生位移”误判成功。
- `control/retry`: 放置失败后的重试逻辑已开始分流：已到位但未释放时优先 `retry release`，已释放但未稳定时优先等待复查，只有真正未到位时才重跑完整 pick-place。
- `control/push`: Push 成功判定已收紧为联合条件：必须沿目标方向取得正向位移，且不能出现明显侧滑、抬高或翻倒。
- `eval/logging`: `execution_summary` 已记录 `planning_source`、`clearing_source`、`grasp_source`、`push_source`、原始视觉目标、校正后视觉目标和执行层诊断，用于区分视觉驱动、observation fallback、执行漂移和物理滑落。

## 当前分组进展

- `vision/perception`: GroundingDINO 接入、缓存脚本、local-only 加载校验和目标/障碍 payload 发布链路已具备。当前薄弱点是视觉目标中心到真实可抓取点的转换仍需要调参或切换到更适合单臂抓取的小方块目标。
- `logic/fsm`: 状态语义保持为 `PLANNING -> CLEARING -> GRASPING`。有障碍时先清障，无障碍时进入抓取；队列陈旧数据已加保护。放置失败后的主流程分流已开始按“未释放 / 未稳定 / 未到位”处理。
- `control/execution`: 单臂抓取和副臂推障均可进入真实控制函数。双臂连续轨迹、放置判定和 release retry 逻辑已作为稳定化成果保留。当前最大问题已从 pick-place 主链转移到 push 动作本身以及日志诊断残留。
- `integration/merge`: `main.py` 仍是集成验证入口。默认路径是单臂抓取 + 单臂清障；双臂共同抓取逻辑保留为实验能力，不作为默认主线。
- `eval/logging`: 最新日志表明 `pick-place` 主链已可跑通，但 `dual_arm_execution_diagnostics` 仍存在残留诊断问题，例如 attempt 已 `success=true` 但 `failed_stage` 仍保留早期 `approach` 假失败。
- `infra/docs`: 依赖版本、任务文档规则和缓存脚本说明已同步；本地环境维护、模型缓存路径和临时运行细节不写入本文件。

## 当前主要问题

### 1. 相机外参仍然是占位值

`configs/camera.yaml` 中的 `T_world_cam` 目前仍然是单位阵占位值，不是真实标定结果。视觉输出的世界坐标仍不是最终可信的物理坐标，目前依赖仿真修正与控制层容错。

### 2. push 动作策略仍然是主失败点

根据最新日志，`pick-place` 主链已经可以完成，但 `post-grasp push test` 仍然经常出现执行偏移：

- 沿错误方向推动；
- 侧滑过大；
- 抬高或顶翻物体；
- 尽管物体发生位移，但不应算成功。

当前瓶颈已经从“能否搬运到目标附近”转移到“push 动作是否按预期方向稳定执行”。

### 3. 连续轨迹控制还不够自然

虽然 `approach / transfer / release` 已经开始连续化，但视觉上仍可能存在：

- 阶段边界停顿；
- 接近抓取窗口时 Z 向收敛偏保守；
- release 和 recheck 之间的等待感较强。

### 4. 日志诊断仍存在残留假失败

`dual_arm_execution_diagnostics` 仍可能出现：

- attempt 最终 `success=true`
- 但 `failed_stage` 仍保留早期 `approach` 假失败

这会让日志诊断与真实任务结果不一致。

### 5. 工程完整度仍然不足

当前系统可以演示主链路，但距离“完整项目”仍有差距：

- 自动化测试覆盖不足；
- 依赖与运行环境仍存在同步风险；
- README / 配置 / 文档对新成员的可复现支持不够强；
- benchmark 场景、评估表和报告模板还没有完全固化。

### 6. RL 尚未进入可落地阶段

项目当前没有使用 RL 训练。当前主线仍应是先把标定、执行稳定性、判定标准和日志基线做扎实，再考虑在局部短时域子任务上引入 RL。

## 今日剩余任务

1. `control/execution`: 继续让连续轨迹控制更自然，减少 `approach` 和 `release` 周期中的视觉停顿。成功标准：主观运动更连贯，且不引入抓取成功率回退。
2. `control/push`: 重构 `execute_push()` 的接触与推进策略。成功标准：沿目标方向取得正向位移，不出现明显推反、顶翻、抬高误判。
3. `eval/logging`: 清理 `dual_arm_execution_diagnostics` 残留假失败。成功标准：attempt 成功时不再残留无效 `failed_stage`。
4. `vision/perception`: 继续调 `single_vision_grasp_offset`；若当前 pot 几何仍不适合单臂抓取，切换到更贴合主线的小方块目标环境。
5. `infra/tests`: 为成功 / 失败判定、push 判定、release retry 分流补最小测试。

## 下一步建议

- 优先调 `single_vision_grasp_offset`，因为这是保持当前感知、FSM、队列和执行接口不变的最小改动。
- 如果 `TwoArmLift` 的 pot 几何继续阻碍单臂视觉抓取，优先切到更适合课程目标的小方块环境，而不是继续硬调 pot 单臂抓取。
- 在默认单臂主线稳定后，再将双臂连续轨迹与更强的放置 / 推障策略作为扩展能力迭代。
- 只有当当前启发式控制与判定逻辑稳定到 benchmark 可复现后，才进入 RL 子任务设计窗口。

## 近期任务

1. 修复 `transformers` 版本要求与依赖文件不一致的问题。
2. 为 `detected_objects`、FSM 状态转移、run log、成功 / 失败判定补最小测试。
3. 提高单臂与双臂连续轨迹执行的速度与观感，减少阶段边界停顿。
4. 优化抓取接近段的闭合窗口和 Z 方向收敛，减少 `approach` 末端保守停顿。
5. 重构 `push` 接触与推进策略，使其通过新的成功判定而不是仅靠位移。
6. 清理连续轨迹重构后的日志残留问题，避免 `dual_arm_execution_diagnostics` 中出现“任务成功但 stage 仍残留 failed_stage”的假失败记录。
7. 重新进行多轮实验，对比修改前后的抓取、放置、推障成功率。

## 中期任务

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
   - push direction progress / lateral drift
7. 整理 RL 前置数据，形成可训练的近场抓取子任务数据集或回放集。

## 后期任务

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

## 2026-04-28 阶段结论

### 关于项目主线

当前项目主线不应转向“大规模模型训练”或“端到端 RL”。更合理的路线仍然是：

- 先把工程基线、标定、执行稳定性和日志评估做扎实；
- 再在局部子任务上引入 RL 进行能力增强；
- 最后视收益情况决定是否扩大 RL 覆盖范围。

### 当前最值得做的事情

- 继续提升连续轨迹执行的速度与观感，减少阶段边界停顿。
- 继续优化抓取接近段的闭合窗口和 Z 方向收敛。
- 优先修复 `push` 动作策略，因为当前主失败已转移到 post-grasp push test。
- 清理连续轨迹重构后的日志残留问题。
- 尽快完成 `T_world_cam` 的真实标定。
- 保持 `vision/`、`ipc/`、`fsm/`、`logs/` 接口稳定，不要为了训练提前改接口。

### 当前执行层补充进展

- 已开始将单臂与双臂 waypoint 执行从离散目标追踪改为平滑参考跟踪。
- 双臂抓取前半段 `transit -> pre_grasp -> align -> final_descent -> grasp_pose` 已开始合并为单次连续 `approach trajectory`。
- 双臂 `transfer / place / release` 主路径已开始从多段循环推进重构为单次连续轨迹跟踪。
- 放置成功判定已收紧为联合条件：目标 XY 到位、物体高度回到桌面附近、姿态保持基本直立、夹爪已释放。
- 放置失败后的重试逻辑已开始分流：已到位但未释放时优先 `retry release`，已释放但未稳定时优先等待复查，只有真正未到位时才重跑完整 pick-place。
- Push 成功判定已收紧为联合条件：必须沿目标方向取得正向位移，且不能出现明显侧滑、抬高或翻倒。
- 最新日志表明：pick-place 主链已可跑通，物体能够被带到目标区域并完成释放，当前主失败点已转移为 `post-grasp push test`。
- 最新日志同时表明：`dual_arm_execution_diagnostics` 仍存在残留诊断问题，需要清理诊断写入逻辑。

### RL 进入窗口的条件

只有在下面条件基本满足后，再进入 RL 开发窗口：

- `detected_objects` 输出稳定，低置信度和无效数据能正确降级；
- `perception_queue -> FSM -> arm execution` 闭环能连续运行；
- `T_world_cam` 已完成真实标定或至少有可量化误差；
- 已积累足够多的成功 / 失败轨迹用于分析；
- 当前启发式控制与判定逻辑已形成稳定 benchmark，可用于公平比较。
