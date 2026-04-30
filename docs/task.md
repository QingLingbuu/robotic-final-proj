# 项目任务总结

> 维护约束：凡是影响项目现状、已完成改动、下一步优先级或剩余任务清单的开发工作，必须同步更新本文件 `docs/task.md`，避免实现进度与任务文档脱节。

## 一、当前状态

当前仓库已经从“混合实验目录”收口成较清晰的工程结构：

- 主业务代码位于 `arm/`、`fsm/`、`vision/`、`rl/`
- 运行时支撑统一收口到 `runtime/`
- 训练、评估、setup、validate 脚本统一归到 `scripts/`
- 外部源码统一 vendoring 到 `third_party/robocasa` 与 `third_party/robosuite`
- RL 输出路径统一固定到 `outputs/rl/`
- 标准上手与环境配置流程以 `docs/getting-started.md` 为准，其他文档和任务交接应避免复制一套平行安装步骤

项目当前仍以 `robosuite` 作为稳定联调基线，同时保留可运行的 `RoboCasa` 单臂 RL smoke 路径。第三档 candidate-driven grasping 主链路已经打通到 contract / planning / logging / RL selector interface 这一层，但候选几何仍是基于目标点的占位实现。

## 二、本轮已完成

- 统一了 runtime bootstrap，不再依赖手工 `PYTHONPATH`
- 清理了 `.sisyphus` 相关项目产物
- 清理了根目录旧 smoke 脚本和旧终端备忘
- 统一了 `third_party/` 布局，并将 `robocasa` / `robosuite` 转为 vendored code
- 清理了 `third_party/` 下的缓存和 `egg-info`
- 收紧了 `.gitignore`，避免继续提交日志、缓存和构建产物
- 将原来的 `infra/`、`ipc/`、`logs/` 三个单文件目录收口为统一的 `runtime/` 包
- 将 `scripts/` 分层为：
  - 顶层主入口：`train_rl.py`、`eval_rl.py`
  - setup：`scripts/setup/`
  - validate：`scripts/validate/`
- 完成第三档抓取链路的第一阶段落地：
  - `vision/detected_objects.py` 支持 `grasp_candidates` schema，且 `grasp_candidates` 成为强制协议字段
  - `vision/perception_loop.py` 的 `top_down` candidate 已从“目标点包装”升级为几何启发式版本：基于顶部高分位层估计顶部中心、抓取高度、朝下朝向、顶部展宽驱动的 `gripper_width`，以及结合顶部覆盖率与表面集中度的候选 `score`
  - `vision/perception_loop.py` 现已为 `cup/mug` 增加几何启发式 `handle_grasp` candidate：从 bbox 局部 3D 点云里提取侧向外凸点集，估计 `P_handle`，并进一步补上几何启发式 `orientation`、基于 handle 点集厚度投影的 `gripper_width`，以及结合外凸强度与把手点占比的候选 `score`，失败时自动回退 `top_down`
  - `fsm/perception_cycle.py` / `fsm/state_machine.py` 已切到 candidate-aware planning，支持选中第一个 candidate 与抓空后切换下一个 candidate
  - `main.py` 单臂抓取路径开始真正消费 `selected_candidate`，candidate source 不再错误叠加旧的单臂 z-offset
  - `runtime/run_logger.py` 已记录 `candidate_count`、`selected_candidate_id`、`selected_candidate_score`、`failure_stage`
  - `rl/contracts.py` / `rl/harness.py` 已提供 selector contract 与 padded selector observation helper
  - 已完成端到端 verification slice：perception payload -> FSM candidate selection / retry -> candidate-driven single-arm target -> run log -> RL selector observation
- 新增和整理文档：
  - [Getting Started](/D:/Code/MyRepositories/robotic-final-proj/docs/getting-started.md)
  - [Docs Index](/D:/Code/MyRepositories/robotic-final-proj/docs/index.md)
  - [RoboCasa Migration Architecture](/D:/Code/MyRepositories/robotic-final-proj/docs/architecture/robocasa-migration.md)
  - [2026-04 History Summary](/D:/Code/MyRepositories/robotic-final-proj/docs/archive/2026-04-history.md)
- 新增 `scripts/demo_vision_grasp.py`，提供独立的视觉抓取演示入口，强制走 GroundingDINO -> `detected_objects` -> `perception_queue` -> FSM candidate selection -> 单臂抓取 这条链路，便于验证“视觉模型参与抓取”的全过程；本轮仅完成静态脚本校验，未做实际仿真运行验证
- 修正 robosuite 视觉几何基线：
  - `arm/env_wrapper.py` 现会将 robosuite 归一化 depth 转换为真实米制深度
  - `arm/env_wrapper.py` 现优先读取 robosuite 运行时相机内外参，而不是仅依赖 `configs/camera.yaml` 占位值
  - `scripts/demo_vision_grasp.py` 与 `main.py` 现会使用运行时相机参数构建视觉坐标变换，避免“检测结果有了，但 3D 抓取点明显偏离物体真实高度”的旧问题
  - `arm/env_wrapper.py` 现统一将 robosuite 默认的 OpenGL 图像 / depth 观测翻转到与 robosuite 官方 camera transform 测试一致的方向，减少像素反投影因图像上下颠倒导致的 3D 点云错位
  - `vision/perception_loop.py` 现支持优先使用 `camera_to_world_transform` 做像素到世界坐标变换，与 robosuite 官方 `transform_from_pixels_to_world` 语义对齐；`scripts/demo_vision_grasp.py` 与 `main.py` 也已把该 4x4 变换注入视觉配置
- 新增 `scripts/demo_robocasa_vision.py`，用于 RoboCasa mug / cup 任务的视觉 smoke；当前已验证 RoboCasa `CoffeeSetupMug` 可起环境、输出 RGB-D、完成 mug 2D 检测，并生成包含 `top_down` / `handle_grasp` 的 3D candidate payload
- `scripts/demo_robocasa_vision.py` 现会额外导出 `png + json` 可视化产物到 `outputs/vision/robocasa_smoke/`，便于人工核对 mug 检测框、target 位置和 grasp candidates；本轮未重新跑仿真，只完成脚本改动与只读 AST 校验
- `scripts/demo_robocasa_vision.py` 现支持 `--execute-reach` 最小执行 smoke：基于视觉选中的 RoboCasa candidate，让机械臂以张开夹爪姿态先移动到候选点上方，再下探到候选点附近，并输出 `robot0_eef_start` / `robot0_eef_final` / `final_error_to_settle` 等摘要；本轮未在当前沙箱里实际跑通环境，只完成脚本实现与只读 AST 校验
- `arm/env_wrapper.py` 与 `scripts/demo_robocasa_vision.py` 现已切到 RoboCasa 运行时相机路径：wrapper 会记录实际命中的 `active_camera_name` / `active_rgb_key` / `active_depth_key`，并优先从底层 sim 读取相机内外参与 `camera_to_world_transform`；这是为修复此前 RoboCasa visual candidate 与 `robot0_eef_pos` 明显不在同一世界坐标系的问题
- `scripts/demo_robocasa_vision.py` 的 `--execute-reach` 现默认先做 RoboCasa 动作轴标定：通过 `action.end_effector_position` 的三轴对称脉冲，估计 “动作轴 -> 世界位移” 3x3 映射矩阵，再用 `pinv` 把世界系 reach 误差反解成动作命令；这是为修复坐标系对齐后仍存在的 RoboCasa 控制轴语义不一致问题
- `scripts/demo_robocasa_vision.py` 现已确认相机采集帧本身正常、黑屏根因在本地视频编码链，因此脚本新增 `--save-frames` / `--save-gif` 作为主可视化导出路径；当前更推荐用逐帧 PNG 或 GIF 排查最后下探轨迹，而不是继续依赖黑屏的本地 mp4/avi 播放链
- 修复 `scripts/demo_robocasa_vision.py` 的 `_resolve_initial_rgbd()` 自递归错误；该 helper 现在会先读取 wrapper 的 `env.get_observation()`，再按 raw observation key 做 RGB / depth 回退，不再在初始 RGB-D 路径上无限递归
- 新增 `scripts/demo_robocasa_reach_onscreen.py`，提供底层 RoboCasa/robosuite 的 onscreen 自动 reach 入口：脚本会同时打开实时 viewer 与 offscreen 相机观测，用运行时 RGB-D 做 candidate 生成，并在窗口里执行可选轴标定和 `hover -> settle` reach，便于直接录屏或人工观察真实机械臂运动
- `requirements.txt` 现合并为统一安装入口，覆盖当前 robosuite、RoboCasa vision 与 RL smoke 路径；该入口采用 RoboCasa/RL 的 `numpy 2.2.5` / `torch 2.7.1` / `torchvision 0.22.1` 基线，并显式安装 `third_party/robosuite` 与 `third_party/robocasa`，不再兼容 `mink 0.0.5` 的 `numpy<2.0.0` 约束
- 清理了本轮试错中确认无效的离屏录像脚本与产物：`scripts/record_robocasa_motion.py`、`scripts/record_robocasa_reach.py` 以及 `outputs/vision/` 下对应 mp4 / reach 调试文件已删除，避免继续误用一条已知会在 `env.step()` 后冻结或黑屏的录像路径
- `third_party/robocasa/robocasa/utils/env_utils.py` 现允许外部覆盖 `use_camera_obs` / `camera_depths`，不再在 `create_env(...)` 内部写死 `camera_depths=False`，为 RoboCasa RGB-D 视觉链打通做准备

## 三、当前主要问题

### 1. `main.py` 仍然偏重

虽然根目录和支撑目录已经明显更干净，但 [main.py](/D:/Code/MyRepositories/robotic-final-proj/main.py) 仍然承担了较重的 orchestration 责任，还不是一个足够薄的主入口。第三档候选选择、抓取执行与日志聚合仍有较多 glue logic 留在这里。

### 2. 第三档候选几何仍是占位实现

当前 `grasp_candidates` 已经贯穿 schema、FSM、execution path、run log 和 RL selector interface，并且两条主候选路径都已摆脱占位实现：`top_down` 与 `cup/mug` 的 `handle_grasp` 都已经具备第一版几何启发式的位姿、夹爪宽度和候选评分。但候选几何仍然不是最终版本：它还缺少更强的多策略候选生成、真实可达性先验，以及比当前启发式更稳的局部形状建模。

### 3. RoboCasa 路径仍有运行时警告

`gymnasium` observation-space warning 仍未收敛。当前 smoke path 已可运行，但还没有把这类 runtime mismatch 彻底收口。

### 4. 测试环境仍受 Windows 权限影响

当前最典型的问题不是 contract 本身，而是：

- `multiprocessing.Queue()` 在部分测试环境下权限拒绝
- 测试里创建 `outputs/` 目录时权限拒绝

这意味着仓库结构已经更整齐，但测试环境还不够干净。

### 5. `third_party/robocasa` 体积较大

当前 vendored 方案换来了“单次 clone 即可工作”，但代价是仓库明显变大。后续如果要进一步优化仓库分发，还需要决定是否继续保留完整资产。

## 四、当前推荐目录职责

- `arm/`: 执行控制与环境适配
- `fsm/`: 状态机与感知消费回合
- `vision/`: 检测、坐标变换、感知循环
- `rl/`: RL contract 与 harness
- `runtime/`: bootstrap、queue、run logger
- `scripts/`: train / eval / setup / validate 入口
- `configs/`: 环境、训练、视觉配置
- `docs/`: 指引、架构、状态板、归档
- `tests/`: contract 与 harness 测试
- `third_party/`: vendored 外部源码

## 五、推荐下一步

- owner group: `vision` + `execution`
- goal: 把 RoboCasa mug 视觉候选从“可检测 / 可导出”推进到“可验证执行”

### 优先级 1

实际运行 `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug`，验证当前机器上“视觉检测 -> candidate -> reach”能否在 onscreen viewer 里稳定展示真实机械臂运动；优先检查 reach 末端是否朝候选点逼近，以及 viewer 中运动与 `reach_summary.final_error_to_settle` 是否一致。

### 优先级 2

若 onscreen reach 成功，再把 RoboCasa 路径从“逼近候选点”推进到“闭合夹爪 + 抬升”两阶段 smoke，并单独记录失败是视觉偏差、控制漂移还是接触后滑脱。

### 优先级 3

继续收薄 [main.py](/D:/Code/MyRepositories/robotic-final-proj/main.py)，把 candidate planning、candidate execution、run log 聚合拆到更稳定的 helper / module 中，减少主入口中的 glue logic。

### 优先级 4

把 `rl/contracts.py` / `rl/harness.py` 的 selector interface 接到真实运行回路，明确：

- selector action 如何选择 candidate
- `clear` / `resense` fallback 如何映射到 FSM
- 训练 / 评估时如何记录 selector 决策质量

### 优先级 5

清理测试环境权限问题，优先解决：

- `multiprocessing.Queue()` 权限拒绝
- 测试创建 `outputs/` 目录权限拒绝

### 优先级 6

在具备 GroundingDINO 缓存和 robosuite 运行环境的机器上实际运行 `python scripts/demo_vision_grasp.py --render`，确认视觉演示入口能稳定产出检测结果、抓取候选和执行结果；若失败，优先记录失败阶段是模型缓存、目标未检出、候选为空还是执行抓取失败。

## 六、建议验证

- `python -m unittest discover -s tests -p "test_*.py"`
- `python -m unittest discover -s tests -p "test_protocol_contracts.py"`
- `python -m unittest discover -s tests -p "test_rl_contracts.py"`
- 端到端 verification slice（本轮已验证）：perception payload -> FSM candidate selection / retry -> candidate-driven single-arm target -> run log -> RL selector observation
- `python scripts/demo_robocasa_vision.py --task robocasa/CoffeeSetupMug --execute-reach`
- `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug`
- `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary`
- `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary`
- `python scripts/demo_vision_grasp.py --render`

## Recorder Failure Handoff (2026-04-29)

- Goal of this branch of work was to produce a RoboCasa mug reach / grasp video directly from code, but that goal was not achieved.
- Direct recorder attempts based on `sim.render(...)`, wrapper cached RGB, raw observation image keys, PNG frame export, GIF export, and local mp4/avi export all showed unstable behavior.
- The most consistent failure pattern was: first frame looked normal, but subsequent frames turned black or near-black after the first `env.step(...)`.
- In the failed recorder experiments, healthy panel stats could appear at stage start, but saved later frames still had mean pixel values near zero; this indicates the failure is upstream of GIF / mp4 encoding and likely tied to RoboCasa observation / render refresh behavior after stepping.
- Additional failure mode observed on later attempts: some runs lost depth at startup (`Depth observation unavailable`), so the reach smoke could not build a 3D payload at all.
- `scripts/demo_robocasa_reach_record.py` was removed because it represented an unverified and misleading path that repeatedly failed in practice.
- The temporary offline `--save-trace-video` branch was also removed from `scripts/demo_robocasa_vision.py` because it was not validated end-to-end and depended on the same unstable reach / depth prerequisites.
- Recommended handoff for the next owner:
  - Treat RoboCasa video capture as an environment / rendering synchronization problem, not an image encoding problem.
  - Start from the stable `scripts/demo_robocasa_vision.py` perception + reach smoke path instead of reviving the removed recorder script.
  - Before any new video work, explicitly compare three sources before and after the first `env.step(...)`: `env.obs`, `env.get_raw_observation()`, and direct `sim.render(...)`.
  - Verify whether depth loss and black-frame loss share the same root cause in the wrapper refresh path.

