# 项目任务进度板

> 维护约束：本文件只记录需要进入 git 仓库的源码、配置、文档、接口状态和下一步任务。不要记录 `.venv` 重建、解释器安装、模型缓存下载、个人机器路径、临时运行日志等本地运行时操作。

## 当前目标

今天的主目标是完成视觉驱动的单臂移动流程：

- `robot0`：根据视觉目标位置抓取并移动目标物体。
- `robot1`：根据视觉障碍物位置执行清障推移。
- `vision -> perception_queue -> FSM -> execution -> logs` 链路保持稳定。
- 视觉模型不可用时可以保留 observation fallback 用于控制调试，但不能把 fallback 误写成视觉链路已完成。

## 最近完成

- `infra/docs`: `AGENTS.md` 已补充 `docs/task.md` 维护协议，要求每次运行前读取、阶段完成后更新、按分组维护进度，并限制 `docs/task.md` 只记录需要进入 git 的源码/配置/文档/接口状态和下一步任务。
- `vision/perception`: 新增 `scripts/cache_grounding_dino.py`，用于按 `configs/vision.yaml` 中的 `model_id` 缓存并校验 GroundingDINO 模型；脚本会验证 `local_files_only=True` 下 processor 和 model 可加载。
- `vision/perception`: `vision/detector.py` 增强本地模型缺失时的错误提示，提示运行缓存脚本，而不是暴露底层加载异常。
- `integration/merge`: 默认任务主线已从“双臂共同抓取同一物体”改回“单臂抓取 + 单臂清障”。`main.py` 默认 `GRASP_MODE = "single"`。
- `integration/merge`: 视觉链路已能驱动真实控制入口：运行中可进入 `planning_source=vision`，单臂阶段可进入 `grasp_source=vision`；如果检测到障碍物，清障阶段使用视觉障碍物位置。
- `logic/fsm`: `fsm/perception_cycle.py` 在发布新视觉 payload 前会清空旧队列，避免清障后读取过期感知结果。
- `control/execution`: 保留远端已有执行层稳定性改动，包括双臂阶段诊断、失败分类、终止 episode 短路和安全撤回结果记录；本轮合并后 `RobosuiteEnvWrapper` 同时维护 `done` 与 `_episode_terminated`，避免 episode 已结束后继续调用 `env.step()` 崩溃。
- `vision/perception`: 单臂视觉抓取偏移已移入 `configs/vision.yaml` 的 `single_vision_grasp_offset`，后续可通过配置调视觉抓取点，不需要改源码。

## 当前分组进展

- `vision/perception`: GroundingDINO 接入、缓存脚本、local-only 加载校验和目标/障碍 payload 发布链路已具备。当前薄弱点是视觉目标中心到真实可抓取点的转换仍需要调参或换成更适合单臂抓取的小方块目标。
- `logic/fsm`: 状态语义保持为 `PLANNING -> CLEARING -> GRASPING`。有障碍时先清障，无障碍时进入抓取；队列陈旧数据已加保护。
- `control/execution`: 单臂抓取和副臂推障均可进入真实控制函数。远端已有双臂执行诊断和失败分类可继续保留，但今天默认主线是单臂移动目标 + 单臂清障。
- `integration/merge`: `main.py` 仍是集成验证入口。双臂共同抓取逻辑保留为实验分支，不作为默认任务路径。
- `eval/logging`: `execution_summary` 已记录 `planning_source`、`clearing_source`、`grasp_source`、`push_source`、原始视觉目标、校正后视觉目标和执行层诊断，用于区分视觉驱动、observation fallback、执行漂移和物理滑落。
- `infra/docs`: 依赖版本和任务文档规则已同步；本地环境维护、模型缓存路径和临时运行细节不写入本文件。

## 今日剩余任务

1. `control/execution`: 调整 `configs/vision.yaml` 中的 `single_vision_grasp_offset`，让视觉目标中心转换到当前物体上更稳定的单臂抓取点。成功标准：`python main.py` 中 `grasp_source=vision` 且单臂抓取阶段返回成功。
2. `integration/merge`: 如果当前 `TwoArmLift` 的 pot 几何继续阻碍单臂抓取，新增或切换到更贴合项目目标的小方块目标环境。成功标准：仍保留双臂系统，`robot0` 负责移动目标小方块，`robot1` 负责清障。
3. `eval/logging`: 完成一次视觉驱动成功运行后，记录源码层面的结论和下一步任务；不要把本地缓存路径、虚拟环境操作或临时日志写入本文档。

## 下一步建议

优先调 `single_vision_grasp_offset`，因为这是最小改动：保持当前感知、FSM、队列和执行接口不变，只修正视觉目标到单臂抓取点的映射。如果调参仍无法成功，应把目标物体切到小方块环境，而不是继续围绕双臂 pot 任务硬调单臂抓取。
