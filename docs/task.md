# 项目任务看板

> 维护约束：凡是影响项目现状、完成工作、下一步优先级、运行结果或交接信息的改动，必须同步更新本文件。历史细节应压缩到 `docs/archive/`，本文件只保留当前可执行状态。

## 当前结论

- 当前稳定基线仍是 `robosuite` + PandaOmron 单臂 top-down 抓取，用它验证 perception / planner / execution 接口。
- `CoffeeSetupMug` 的 top-down 路径已被用户手动验证可 reach / close / lift；当前最终 demo 主线收敛为 top-down。
- handle side / oblique 抓取保留为局限性展示路径；允许在 handle 展示中接触甚至打翻杯子，但最终方案仍使用 top-down。
- 已撤销：`handle_guided_top_down` 中间展示路径。实际运行显示它会引入语义/执行误解，当前不再保留为 demo 模式。
- 最终 base 策略：只有保守小步移动 base；base 发生有效移动后默认重新渲染并重新跑 vision，使用刷新后的 candidate 再 reach，避免沿用移动前的旧抓取点。
- 本分支目标是先把混乱 demo 拆成可审查模块，并加入诊断工具；暂不把 handle 抓取设为默认成功路径。
- 旧版长任务记录已归档到 `docs/archive/task-2026-05-01-pre-cleanup.md`，当前文件只保留短看板。

## 分组状态

### vision/perception

- 已完成：`vision/perception_loop.py` 能输出 `top_down`、`handle_grasp` 与 `handle_top_down` candidate，demo 继续通过 `VisionPerceptionLoop` 生成 `detected_objects`。
- 保持不变：`detected_objects` schema 与 `perception_queue` 协议未在本轮重构中修改。
- 已完成：`handle_top_down` 点云入口增加 bbox 内前景深度连通过滤，用明显深度断点排除背景 / 台面点；把手估计继续采用“先建模杯身、再用残差簇识别把手”的全局点云方法，并保留 sim mesh 只做验证对照。
- 已修正：`handle_top_down` 的 closing axis 现在使用把手 outward / radial 方向，避免夹爪沿把手切向平行闭合；planner fallback 从 `handle_grasp` 合成 `handle_top_down` 时也使用同一语义。
- 风险：handle candidate 的几何位置已通过 sim mesh 对照降到毫米级误差，但接触高度 / 闭合宽度仍需继续用真实 RoboCasa 执行验证。

### control/execution

- 已完成：从 `scripts/demo_robocasa_reach_onscreen.py` 抽出执行 helpers 到 `arm/robocasa_execution.py`。
- 已完成：抽出轴向标定到 `arm/calibration.py`，抽出 top-down reach / close / lift 到 `arm/robocasa_primitives.py`。
- 已完成：抽出 handle / oblique 实验执行到 `arm/handle_experiments.py`，并通过 `--handle-mode` 显式启用。
- 已完成：新增 `arm/action_space_diagnostics.py`，demo 支持 `--diagnose-action-space` 扫描 action index 对 EEF / gripper 的影响。
- 已完成：新增 `arm/reachability.py`，demo 输出 `reachability_diagnosis`，用于区分 near-success、position saturation、orientation saturation、gripper width risk。
- 已完成：新增实验性 `arm/base_torso.py`，demo 支持 `--enable-base-torso-preposition` 在 arm reach 前执行 base 预定位；默认关闭，torso 默认禁用，避免污染稳定 top-down baseline。
- 已完成：base 预定位已参数化，demo 支持 `--base-desired-xy-standoff`、`--base-xy-deadband`、`--base-preposition-gain`、`--base-action-limit`，便于强制触发和验证 base movement。
- 已完成：新增 `--auto-preposition-retry`，top-down 首次 reach 失败且诊断显示 position saturation 或 final error 过大时，会执行一次 base preposition 并 retry reach。
- 已完成：新增 base 专用 action mapping 标定，demo 支持 `--calibrate-base-action-mapping` 输出 base action 到 EEF/Base XY 的实测映射，支持 `--use-base-action-mapping` 让预定位用该映射反解 base action。
- 已完成：`scripts/demo_robocasa_reach_onscreen.py` 在 base 预定位发生有效移动后默认刷新 vision candidate；仅调试时可用 `--skip-vision-refresh-after-base-preposition` 保留旧 candidate。
- 已完成：handle side / oblique 路径明确为 reach-only 验证，不进入 close/lift；默认允许继续到 contact approach，用于展示 handle reachability / 姿态控制局限性。
- 已完成：新增 `handle_top_down` 实验路径，执行时在 hover 后先对齐 candidate orientation，再下探 close/lift；普通 `top_down` 执行保持不对齐姿态，避免污染稳定杯口 baseline。
- 已修复：`handle_top_down` 的动态把手方向现在按 Panda 夹爪实际闭合轴（EEF Y 轴）对齐；此前 perception 已输出随把手变化的 `closing_axis`，但 execution/debug 按 EEF X 轴解释，导致视觉上像每次重启都是同一方向。
- 已撤销：handle base 预定位目标改为 pre-approach target 的实验；该改动导致模型卡在预靠近阶段。当前 base preposition 已恢复为对齐 selected candidate 本身。

### logic/fsm

- 已完成：demo 的 candidate 策略从脚本迁移到 `planner/candidates.py`。
- 当前策略：`--handle-mode off` 为默认，handle candidate 会回退到稳定 top-down；`anchor_top_down`、`oblique_reach`、`side_reach` 为显式实验模式；`--grasp-type handle_top_down` 可显式选择从上方抓把手。
- 保持不变：正式 FSM 状态迁移和 `detected_objects` 消费协议未在本轮改动。

### integration/merge

- 当前分支：`fix-execution-cleanup-baseline`。
- 改动范围：`arm/`、`planner/`、`scripts/demo_robocasa_reach_onscreen.py`、`tests/`、`docs/task.md`。
- 未触碰：`main.py`、正式 `fsm/` IPC 协议、`vision/detected_objects.py` schema。

### eval/logging

- 本轮未新增正式 run log schema；只在 demo stdout 中加入 `action_space_diagnostics` 与 `reachability_diagnosis` JSON。
- 用户运行证据：`CoffeeSetupMug + top_down` 曾达到 `overall_success=true`；handle / oblique 仍失败，日志显示 position / orientation action 饱和与 workspace 边界问题。
- 用户运行证据：`CoffeeSetupMug + top_down + --enable-base-torso-preposition` 多次达到 `overall_success=true`，说明预定位开关未破坏稳定路径；最近一次输出 `effective_should_move=false`、`action_norm=0.0`、`torso_command_ignored=true`，验证了“规划想动 torso 但 torso 默认禁用，因此实际不移动”的 summary 语义。
- 用户运行证据：激进 base 参数 `--base-desired-xy-standoff 0 --base-xy-deadband 0 --base-preposition-gain 8 --base-action-limit 0.5` 能触发 `effective_should_move=true` 与 `action_norm>0`，EEF 发生明显 XY 位移，且 top-down 仍 `overall_success=true`；但该次 `xy_distance_improved=false`，说明当前 base command 方向还需要 base 专用标定，不能复用 arm action mapping 当作 base mapping。
- 用户运行证据：`--auto-preposition-retry` 在 top-down 首次 reach 成功时输出 `triggered=false`，符合“不干扰成功 baseline”的预期。
- 用户运行证据：首次 `--calibrate-base-action-mapping --use-base-action-mapping` 暴露出两个问题：`action[10]` 主要影响 Z / height，不应放进 XY base slice；mapping 反解还需要按执行步数缩放，否则 30 步会严重过冲。默认 base slice 已收紧为 `7:10`。
- 用户运行证据：`base_slice=[7,10]` 后 mapping 输出维度正确且 top-down 仍成功，但强制预定位仍出现 `xy_distance_improved=false` 与 Y 方向过冲；已新增 `--base-mapping-trust` 和 `--base-max-world-delta` 作为保守安全缩放，默认只信任 mapping 25% 且每次目标世界位移最多 3cm。
- 用户运行证据：保守 mapping 参数 `--base-action-limit 0.2 --base-mapping-trust 0.25 --base-max-world-delta 0.03` 验证有效：`base_mapping_used=true`、`action_norm≈0.0285`、`xy_distance_before≈0.1457`、`xy_distance_after≈0.1366`、`xy_distance_improved=true`，且 top-down reach / close / lift 均成功。
- 用户运行证据：最终路径 `CoffeeSetupMug + top_down + conservative base mapping + vision refresh` 已验证成功；输出包含 `candidate_after_base_preposition`，刷新后 candidate 从 `[3.75878, -0.56763, 0.99767]` 更新到 `[3.75812, -0.56728, 0.99768]`，`execution_summary.candidate` 使用刷新后的 candidate，`xy_distance_improved=true`，`overall_success=true`。
- 用户运行证据：`handle_grasp + oblique_reach` 会刷新 handle candidate 且 base XY 距离改善，但接触 approach 阶段位置 action 饱和并出现 0.62m 级 position drift，机械臂晃动并打翻杯子；因此 handle reach-only 已改为默认非接触验证，只有显式 `--allow-handle-contact-approach` 才允许进入接触 approach。
- 用户运行证据：handle 非接触安全版可避免打翻杯子，但用户接受 handle 展示中打翻杯子以体现局限性；随后 pre-approach base target 实验造成预靠近卡住，已按用户要求撤销。
- 决策：最终演示和后续评估优先使用 top-down；handle 不再作为近期主线。
- 下一轮如进行批量实验，应把诊断字段写入正式 eval log，而不是只依赖终端输出。

### infra/docs

- 已完成：本文件压缩为当前任务看板，避免历史追加项掩盖下一步。
- 已完成：重写前的旧版 `docs/task.md` 已保留到 `docs/archive/task-2026-05-01-pre-cleanup.md`。
- 保留：旧详细背景可继续放入 `docs/archive/` 或独立 design plan；不要再在本看板无限追加长日志。

## 最近完成阶段

- `planner/candidates.py`：集中 candidate 选择、handle fallback、anchor top-down、oblique candidate 构造。
- 已撤销：`handle_guided_top_down` 及其在 `arm/robocasa_primitives.py` 中引入的通用 candidate-orientation 消费；最终路线回到纯 top-down baseline。
- `arm/robocasa_execution.py`：集中 action 构造、EEF pose / quat、夹爪宽度读取。
- `arm/calibration.py`：集中 action mapping 标定与 base-to-world rotation 提取。
- `arm/robocasa_primitives.py`：集中 top-down reach、close、lift，并新增 `handle_top_down` 使用的 oriented top-down reach。
- `arm/handle_experiments.py`：集中 handle side / oblique reach-only 实验逻辑。
- `arm/action_space_diagnostics.py`：新增 action index 扫描诊断。
- `arm/reachability.py`：新增 reach summary 归因诊断。
- `arm/base_torso.py`：新增 base / torso action 构造、预定位命令计算、预定位执行 helper。
- `arm/base_torso.py`：新增 `calibrate_base_action_mapping()`，只 pulse base action slice 并估计 `eef_xy_delta_from_base_action` / `base_xy_delta_from_base_action`。
- `scripts/demo_robocasa_reach_onscreen.py`：新增 base 预定位调试参数，并在 summary 中报告 `xy_distance_before`、`xy_distance_after`、`xy_distance_improved`。
- `scripts/demo_robocasa_reach_onscreen.py`：base 有效移动后默认输出 `candidate_after_base_preposition`，并让后续 reach / close / lift 使用刷新后的 vision 抓取点。
- `arm/reach_retry.py`：新增 reach 失败后是否触发 base preposition retry 的最小策略判断。
- `arm/handle_validation.py`：新增轻量 handle reach-only 稳定性汇总，避免测试导入重型 robosuite 模块。
- `arm/handle_phases.py`：新增轻量 handle phase 规划；已移除 handle preposition target helper。
- `arm/handle_experiments.py`：side / oblique handle reach summary 现在带 `reach_only=true` 与 `handle_reach_validation`；默认会继续到 contact approach，`--handle-noncontact-only` / `--handle-abort-on-position-drift` 可用于安全调试。
- `vision/perception_loop.py`：新增 `handle_top_down` candidate，使用把手点云顶部高度、垂直 outward 的 closing axis、把手厚度投影作为 gripper width。
- `scripts/demo_robocasa_reach_onscreen.py`：`--grasp-type` 支持 `handle_top_down`，该路径走 orientation-aware top-down reach 后继续 close/lift。
- `arm/robocasa_primitives.py` / `scripts/demo_robocasa_reach_onscreen.py`：修正 `handle_top_down` 姿态消费约定，让 candidate `closing_axis` 对齐当前 EEF Y 轴，并让 yaw debug 使用同一个闭合轴定义。

## 验证记录

- 已通过：`python -m unittest discover -s tests -p "test_candidate_planner.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_robocasa_execution_helpers.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_calibration_helpers.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_action_space_diagnostics.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_base_torso.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_reachability_diagnosis.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_reach_retry.py"`。
- 已通过：`python -c "from pathlib import Path; files=['scripts/demo_robocasa_reach_onscreen.py','arm/reach_retry.py','arm/base_torso.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过：`python -m unittest discover -s tests -p "test_base_torso.py"`（vision-refresh 接入后复验）。
- 已通过：`python -m unittest discover -s tests -p "test_reach_retry.py"`（vision-refresh 接入后复验）。
- 已通过：`python -c "from pathlib import Path; files=['scripts/demo_robocasa_reach_onscreen.py','arm/base_torso.py','arm/reach_retry.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`（vision-refresh 接入后复验）。
- 已通过：`python -c "from pathlib import Path; files=['scripts/demo_robocasa_reach_onscreen.py','arm/action_space_diagnostics.py','arm/base_torso.py','arm/reachability.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过：`python -c "from pathlib import Path; files=['scripts/demo_robocasa_reach_onscreen.py','arm/robocasa_primitives.py','arm/handle_experiments.py','arm/action_space_diagnostics.py','arm/reachability.py','planner/candidates.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过用户手动仿真验证：`python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug --grasp-type top_down --enable-base-torso-preposition --calibrate-base-action-mapping --use-base-action-mapping --base-desired-xy-standoff 0 --base-xy-deadband 0 --base-preposition-gain 1 --base-action-limit 0.2 --base-mapping-trust 0.25 --base-max-world-delta 0.03 --render-sleep-sec 0 --keep-open-sec 1`，结果 `overall_success=true`。
- 已通过：`python -m unittest discover -s tests -p "test_handle_experiments.py"`。
- 已通过：`python -c "from pathlib import Path; files=['arm/handle_validation.py','arm/handle_experiments.py','scripts/demo_robocasa_reach_onscreen.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过：`python -m unittest discover -s tests -p "test_handle_experiments.py"`（handle 非接触 phase 复验）。
- 已通过：`python -c "from pathlib import Path; files=['arm/handle_phases.py','arm/handle_validation.py','arm/handle_experiments.py','scripts/demo_robocasa_reach_onscreen.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已修复：`arm/handle_experiments.py` 从 `arm/handle_phases.py` 导入 `HANDLE_STANDOFF`，解决 handle 脚本启动时报 `NameError: HANDLE_STANDOFF is not defined`；同时 abort 后不再继续后续 handle phase。
- 已通过：`python -m unittest discover -s tests -p "test_handle_experiments.py"`（handle 默认 contact limitation demo 复验）。
- 已通过：`python -c "from pathlib import Path; files=['arm/handle_phases.py','arm/handle_validation.py','arm/handle_experiments.py','scripts/demo_robocasa_reach_onscreen.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`（handle 默认 contact limitation demo 复验）。
- 已通过：`python -m unittest discover -s tests -p "test_candidate_planner.py"`（撤销 `handle_guided_top_down` 后复验）。
- 已通过：`python -m unittest discover -s tests -p "test_robocasa_execution_helpers.py"`（撤销 candidate-orientation 消费后复验）。
- 已通过：`python -m unittest discover -s tests -p "test_handle_top_down_candidate.py"`。
- 已通过：`python -m unittest discover -s tests -p "test_robocasa_execution_helpers.py"`（新增 oriented top-down rotation helper 后复验）。
- 已通过：`python -m unittest discover -s tests -p "test_candidate_planner.py"`。
- 已通过：`python -c "from pathlib import Path; files=['vision/perception_loop.py','arm/robocasa_primitives.py','scripts/demo_robocasa_reach_onscreen.py','tests/test_handle_top_down_candidate.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过：`python -c "from pathlib import Path; files=['arm/robocasa_primitives.py','scripts/demo_robocasa_reach_onscreen.py','tests/test_robocasa_execution_helpers.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`（`handle_top_down` 闭合轴修正后语法校验）。
- 未完成：`python -m unittest discover -s tests -p "test_robocasa_execution_helpers.py"` 在本轮被用户中断；需要后续重跑确认完整 helper 单测。

## 推荐下一步

- owner group: `control/execution`
- goal: 手动验证 `handle_top_down` 的夹爪闭合方向已从切向改为 outward/radial，且下调后的接触高度能稳定夹住把手。
- success criteria: `target_closing_dot_outward` 接近 1 或 -1，而不是接近 0；`final_closing_dot_target_closing` 绝对值接近 1；`object_gripper_contact=true` 且 lift 阶段物体 z 有上升。
- suggested verification: 运行 `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/HeatMug --target-label mug --grasp-type handle_top_down --layout 3 --keep-open-sec 1`，观察夹爪是否跨把手厚度闭合，而不是沿把手方向闭合。
