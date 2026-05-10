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
- 已完成：新增 `robocasa/CupMugSortingRandom`，复用 5 物体 cup/mug sorting 逻辑，但不再默认固定左水槽 `layout_and_style_ids: [[1, 1]]`，用于随机厨房场景可视化与对比实验。

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
- 已完成：新增 `scripts/export_vision_pointclouds.py`，可把单帧 RoboCasa RGB-D 导出为 `outputs/vision/...` 下的 RGB、depth、检测 overlay、点云 `.ply`、三平面点云投影视图，以及相机视角点云重投影 / 抓取轴叠加图，便于课程展示和视觉调试。
- 已完成：`scripts/export_vision_pointclouds.py` 现支持 `CupMugSorting` 多目标导出模式；当 `--target-label` 取 `all` / `drinkware` 时，会在同一套图里同时绘制 cup 与 mug 的点云、候选抓取点和抓取轴。
- 已完成：`scripts/export_vision_pointclouds.py` 现在对 `CupMugSorting` / `CupMugSortingClean` 自动复用固定左水槽布局，显式传入 `layout_and_style_ids: [[1, 1]]`，避免导出脚本采样到与演示不一致的场景。
- 已完成：`scripts/export_vision_pointclouds.py` 的环境创建链路已切回与 `scripts/demo_robocasa_reach_onscreen.py` 一致的 `build_env_config() + robosuite.make(...) + get_rgbd()/resolve_camera_config()`，不再使用 `RobocasaEnvWrapper/gym.make` 路径，避免“任务名相同但场景不一致”。
- 已完成：多目标导出模式现保留原始 DINO bbox 图 `rgb_detection_overlay.png`，并把 cup/mug 抓取叠加单独输出为 `rgb_multi_target_overlay.png`，避免检测结果与抓取结果混在一张图里。
- 已完成：新增 `configs/tasks/cup_mug_sorting_random.yaml`，并更新相关脚本使 `CupMugSortingRandom` 走同一套 cup/mug 多目标视觉流程，但默认不固定 layout/style。
- 已完成：从 `outputs/vision/cup_mug_sorting_all/` 挑选一套代表性可视化产物归档到受版本控制的 `docs/artifacts/vision/cup_mug_sorting_all/`，用于评审、汇报和主分支留存，同时保持 `outputs/` 继续作为运行时产物目录。
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
- `scripts/export_vision_pointclouds.py`：新增单帧视觉可视化导出工具，直接复用 `VisionPerceptionLoop` 的真实 bbox 点云和 candidate，输出 `scene/target` 点云三视图、相机平面重投影图、RGB 抓取点与抓取轴叠加图，以及 `.ply` 点云文件。
- `scripts/export_vision_pointclouds.py`：新增 `CupMugSorting` 多目标模式，复用 `infer_all_targets_with_diagnostics()` 与 `classify_drinkware_targets()`，把 cup 和 mug 同时画进同一张 overlay / 重投影图里。
- `scripts/export_vision_pointclouds.py`：环境初始化与相机取图现复用 onscreen reach demo 的同一路径，确保视觉导出面对的是同一套 RoboCasa/robosuite 场景配置。
- `third_party/robocasa/.../cup_mug_sorting.py`：新增 `CupMugSortingRandom` 类；`scripts/demo_robocasa_reach_onscreen.py`、`scripts/demo_multi_cup_mug_sort.py`、`scripts/export_vision_pointclouds.py` 已支持该任务名，并保持随机版不自动固定左水槽布局。

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
- 已通过：`python -c "from pathlib import Path; files=['scripts/export_vision_pointclouds.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 已通过：`python -c "from pathlib import Path; files=['third_party/robocasa/robocasa/environments/kitchen/composite/organizing_dishes_and_containers/cup_mug_sorting.py','third_party/robocasa/robocasa/__init__.py','scripts/demo_robocasa_reach_onscreen.py','scripts/demo_multi_cup_mug_sort.py','scripts/export_vision_pointclouds.py','tests/test_cup_mug_sorting_scene.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"`。
- 未完成：`python -m unittest discover -s tests -p "test_robocasa_execution_helpers.py"` 在本轮被用户中断；需要后续重跑确认完整 helper 单测。

## 推荐下一步

- owner group: `control/execution`
- goal: 手动验证 `handle_top_down` 的夹爪闭合方向已从切向改为 outward/radial，且下调后的接触高度能稳定夹住把手。
- success criteria: `target_closing_dot_outward` 接近 1 或 -1，而不是接近 0；`final_closing_dot_target_closing` 绝对值接近 1；`object_gripper_contact=true` 且 lift 阶段物体 z 有上升。
- suggested verification: 运行 `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/HeatMug --target-label mug --grasp-type handle_top_down --layout 3 --keep-open-sec 1`，观察夹爪是否跨把手厚度闭合，而不是沿把手方向闭合。
## 2026-05-02 Cup/Mug Sorting Environment Update

- Completed: added RoboCasa composite task `CupMugSorting` with multiple mugs and cups placed on one counter. Initial placement intentionally interleaves handled mugs and plain cups, so the sort check is not already satisfied at reset.
- Completed: added `CupMugSortingClean`, a two-object clean validation scene with one mug and one plain cup in a front-facing open row. Use this before returning to the denser multi-object scene.
- Completed: plain no-handle cups are now constrained to top-down-feasible geometry (`object_scale=[0.75, 0.75, 1.0]`, max XY size 7.2cm). This is required because Panda top-down cannot lift a cup whose rim is as wide as the maximum gripper opening.
- Completed: added `configs/tasks/cup_mug_sorting.yaml` to track the environment target labels, object counts, sorting zones, and expected grasp strategies.
- Completed: added `VisionPerceptionLoop.infer_all_targets_with_diagnostics()` for smoke-test multi-object perception without changing the existing single-target `detected_objects` queue schema.
- Completed: added `vision/sorting_policy.py` and `scripts/demo_multi_cup_mug_sort.py` to classify visible drinkware. Objects with a `handle_top_down` candidate are assigned to the right arm / handle path; plain top-down-only objects are assigned to the left arm / top-down path.
- Verification: syntax check passed for the new environment, script, config, perception, policy, and test files. The focused unittest command timed out in this local environment after importing robosuite warnings, so full simulation smoke verification still needs to be run manually.
- Recommended next step: owner group `integration/merge`; goal: run `python scripts/demo_multi_cup_mug_sort.py --task robocasa/CupMugSorting --layout 3 --keep-open-sec 1`; success criteria: env initializes, `cup_mug_sorting_summary.target_count` is greater than 1 when detector cache is available, and assignments split handled mugs to `handle_top_down` and plain cups to `top_down`.

## 2026-05-02 Top-Down Cup Grasp Tuning

- Completed: cup-like top-down candidates now report a wide gripper target (`cup_like_top_down_width_min: 0.075`, capped by `top_down_width_max: 0.08`) so cup/mug top-down attempts are treated as near-max opening instead of inheriting a too-small point-cloud short-axis estimate.
- Completed: cup-like top-down candidates now target below the rim (`cup_like_top_down_penetration_offset: 0.035`) so the close phase grasps cup wall / cup body instead of closing at the rim where max gripper width equals cup opening width.
- Completed: regular top-down final settle descent is more conservative (`TOP_DOWN_SETTLE_ACTION_SCALE = 0.25`); `handle_top_down` keeps its existing settle scale to avoid perturbing the handle path.
- Verification: syntax/config checks passed. Focused Python test commands still time out during robosuite initialization in this local shell, so manual run evidence is required.
- Suggested verification: run `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --layout 3 --keep-open-sec 5` and inspect `perception_candidates[*].gripper_width`, `reach.history_tail`, `close.gripper_width_before`, and `lift.object_gripper_contact`.

## 2026-05-02 Cup/Mug Classified Target Selection

- Completed: `scripts/demo_robocasa_reach_onscreen.py` now enables drinkware classification for `CupMugSorting` when the requested target is `cup`, `glass cup`, or `mug`. It detects all drinkware labels first, classifies by handle availability, then rebuilds the single-target payload from the selected classified object.
- Completed: `vision/sorting_policy.py` now exposes `select_drinkware_target()`. `cup` requests prefer no-handle targets even if a handled mug has higher detector confidence; `mug` requests prefer handle targets.
- Completed: low-score `handle_top_down` candidates are no longer enough to mark an object as handled. The current policy requires either `handle_top_down` score >= 0.30 or strong geometry evidence (`handle_point_count >= 24` and `protrusion_quality >= 0.45`). This prevents weak false positives while still catching borderline-score handled objects.
- Completed: for the synthetic `CupMugSorting` environment, visual targets are now matched to the nearest sim object metadata (`cup_*` / `mug_*`) before classification. This keeps DINO label noise and false visual handle candidates from overriding known scene truth.
- Completed: when the requested `CupMugSorting` class is present in sim metadata but not visible to DINO, the demo now builds a sim-metadata fallback target instead of aborting. This is mainly for `cup` runs where the camera only sees mugs but the environment contains `cup_*` objects.
- Verification: `python -m unittest discover -s tests -p "test_sorting_policy.py"` passed; syntax check passed for `vision/sorting_policy.py` and `scripts/demo_robocasa_reach_onscreen.py`.
- Suggested verification: rerun `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --layout 3 --keep-open-sec 5` and inspect `drinkware_classification.selected_assignment.has_handle=false`.

## 2026-05-02 Cup/Mug Post-Lift Placement

- Completed: added `planner/sorting_zones.py` to choose world-space placement targets for `CupMugSorting`: handled mugs go left of the current object set, plain cups go right.
- Completed: added `execute_place()` to `arm/robocasa_primitives.py`. After lift it transfers above the target zone, descends to release height, opens the gripper, and retracts.
- Completed: `scripts/demo_robocasa_reach_onscreen.py` now runs place after lift for classified `CupMugSorting` targets and includes `place_plan` / `place` in `execution_summary`.
- Verification: `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed; syntax check passed for `arm/robocasa_primitives.py`, `scripts/demo_robocasa_reach_onscreen.py`, and `planner/sorting_zones.py`.
- Suggested verification: run one plain cup path and one handled mug path. For cup: `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --layout 1 --keep-open-sec 5`. For mug: `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label mug --grasp-type handle_top_down --layout 1 --keep-open-sec 5`. Check `execution_summary.place_plan.zone` is `plain` for cup and `handled` for mug.

## 2026-05-02 Fixed Left-Sink Layout

- Completed: `CupMugSorting` and `CupMugSortingClean` now register the kitchen sink and bind object placement to the counter referenced by that sink, instead of choosing an arbitrary cabinet-adjacent counter.
- Completed: sorting demo scripts default Cup/Mug tasks to RoboCasa `layout 1` when `--layout` is omitted. `layout001.yaml` places the sink on the left side of the main counter.
- Completed: task configs now record `fixed_layout.layout_id: 1` and update verification commands to use the fixed left-sink layout.
- Verification: syntax check passed for `cup_mug_sorting.py`, `demo_multi_cup_mug_sort.py`, and `demo_robocasa_reach_onscreen.py`; YAML parsing passed for both Cup/Mug task configs. `test_cup_mug_sorting_scene.py` still times out in this shell during robosuite import, matching previous local behavior.
- Suggested verification: run `python scripts/demo_multi_cup_mug_sort.py --task robocasa/CupMugSortingClean --keep-open-sec 5` and confirm the printed env config has `"layout_ids": 1` and the viewer shows the left-side sink scene.

## 2026-05-02 Strict Cup/Mug Scene Pinning

- Completed: Cup/Mug demo scripts now pass RoboCasa `layout_and_style_ids: [[1, 1]]` for the default fixed scene instead of separately passing `layout_ids=1` and `style_ids=1`.
- Rationale: RoboCasa samples from `layout_and_style_ids` during model setup; passing one explicit pair removes any ambiguity about layout/style expansion while still leaving `seed` unset so cup/mug instances and positions can vary.
- Verification: syntax check passed for both demo scripts. Import-based config introspection still times out in this shell during robosuite initialization, so verify through the printed JSON from the viewer command.
- Suggested verification: run `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --keep-open-sec 5` and confirm the printed config shows `"layout_ids": null`, `"style_ids": null`, and `"layout_and_style_ids": [[1, 1]]`.

## 2026-05-02 Grasp Path Cleanup

- Completed: removed the failed reach-only handle experiment paths from the active codebase: `oblique_reach`, `side_reach`, `anchor_top_down`, `handle_oblique_grasp`, and `handle_anchor_top_down`.
- Completed: deleted dead experiment modules `arm/handle_experiments.py`, `arm/handle_phases.py`, `arm/handle_validation.py`, and `tests/test_handle_experiments.py`.
- Completed: `scripts/demo_robocasa_reach_onscreen.py` now exposes only `top_down`, `handle_top_down`, and `any`; default candidate generation in configs/scripts is limited to `top_down` and `handle_top_down`.
- Kept intentionally: internal `handle_grasp` candidate support in `vision/perception_loop.py` and `planner/candidates.py` remains only as a fallback source for synthesizing `handle_top_down` when direct handle-top-down is unavailable.
- Verification: code search confirms the removed experiment keywords are gone from active `arm/`, `planner/`, `scripts/`, `tests/`, `configs/`, and `vision/` paths. Suggested verification: run `python -m unittest discover -s tests -p "test_candidate_planner.py"` plus the Cup/Mug onscreen cup and mug commands.

## 2026-05-02 Getting Started CJY Docs

- Completed: renamed `docs/getting-started.md` to `docs/getting_start_cjy.md` and updated `docs/index.md`.
- Completed: expanded the getting-started guide with current `CupMugSorting` commands, including onscreen cup/mug runs, scene smoke, frame saving, expected JSON fields, fixed `layout_and_style_ids: [[1, 1]]`, and the active grasp path list.
- Verification: confirmed the old path no longer exists, the new path exists, and `docs/index.md` links to `docs/getting_start_cjy.md`.

## 2026-05-02 RoboCasa Asset Ignore Policy

- Completed: kept the large RoboCasa asset tree out of Git by restoring the asset ignore rules in `third_party/robocasa/.gitignore`.
- Verification: asset files staged by the interrupted add attempt were removed from the index with `git restore --staged third_party/robocasa/robocasa/models/assets`. `git check-ignore -v` again reports ignore rules for representative `objects`, `textures`, `generative_textures`, and fixture asset paths.
- Recommended next step: owner group `infra/docs`; goal: document how teammates should obtain local RoboCasa assets without committing the 22GB local asset tree; success criteria: custom `CupMugSorting` code remains tracked while large third-party assets remain local or are managed by an explicit external asset workflow.

## 2026-05-03 RL Cup/Mug Ordering Phase 1 Contract

- Completed: added `configs/rl/cup_mug_ordering_robocasa.yaml` for Phase-1 high-level RL ordering with `mode: cup_ordering`, `max_targets: 5`, `action_space: Discrete(5)`, `task_name: robocasa/CupMugSorting`, `layout_and_style_ids: [[1, 1]]`, `num_mugs: 2`, and `num_cups: 3`.
- Completed: extended `rl/contracts.py` with a separate `cup_ordering` validation and summary branch, preserving the legacy continuous-action PPO config contract for existing RoboCasa / robosuite configs.
- Completed: expanded `tests/test_rl_contracts.py` to assert the new cup-ordering summary fields and `scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary` output.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed; cup-ordering train summary and legacy single-arm train summary both exited `0`; modified Python files compile with `syntax ok`.
- Note: `lsp_diagnostics` could not run because `basedpyright-langserver` is not installed in this local environment.
- Recommended next step: owner group `logic/fsm`; goal: implement the shared 5-slot cup/mug observation and action-mask builder in `rl/cup_mug_observation.py`; success criteria: `tests/test_cup_mug_ordering_contracts.py` covers exactly 5 slots, stable `mug_1`, `mug_2`, `cup_1`, `cup_2`, `cup_3` ordering, padded/finished/low-confidence/unknown masks, and all-invalid behavior; suggested verification: run `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"`.

## 2026-05-03 RL Cup/Mug Ordering Phase 1 Observation

- Completed: added `rl/cup_mug_observation.py` with pure 5-slot observation/action-mask helpers for `CupMugSorting`; slot order is explicitly stabilized as `mug_1`, `mug_2`, `cup_1`, `cup_2`, `cup_3` when sim object names are available, with deterministic fallback ordering when names are missing.
- Completed: observation slots now expose `visible`, `finished`, `has_handle`, `type_id`, `conf`, `pos`, `candidate_score`, `reachability`, `place_zone_id`, `retry_count`, `valid_action`, and `recommended_grasp`; padded/finished/low-confidence/unknown/duplicate/invisible slots are masked invalid without mutating `detected_objects`.
- Completed: added `tests/test_cup_mug_ordering_contracts.py` covering all-valid five-object scenes, zero/partial target padding, more-than-five truncation, duplicate object ids, low confidence, unknown metadata, finished slots, invisible slots, fallback ordering, and global step context.
- Verification: `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"` passed; `python -m unittest discover -s tests -p "test_sorting_policy.py"` passed; `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed; Task-2 Python files compile with `syntax ok`.
- Known blocker: `python -m unittest discover -s tests -p "test_protocol_contracts.py"` currently fails in `test_infer_detected_objects_adds_handle_grasp_for_cup_like_target` with `AssertionError: 'handle_grasp' not found in ['top_down']`; this predates the new RL observation module and must be fixed before final verification can claim protocol contracts clean.
- Note: `lsp_diagnostics` could not run because `basedpyright-langserver` is not installed in this local environment.
- Recommended next step: owner group `logic/fsm`; goal: add shared random and deterministic risk-aware greedy ordering policies in `rl/cup_mug_policies.py`; success criteria: random samples only valid actions, greedy ranks by candidate score/conf/reachability/retry/slot index deterministically, and all-invalid returns a controlled no-op; suggested verification: run `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"`.

## 2026-05-03 RL Cup/Mug Ordering Phase 1 Baselines

- Completed: added `rl/cup_mug_policies.py` with shared `select_action()` support for `random` and deterministic risk-aware `greedy` policies; both consume the existing observation/action mask and do not duplicate observation-building logic.
- Completed: random policy samples only valid action indices and returns valid-action debug metadata; greedy ranks valid slots by candidate score, confidence, reachability, retry count, distance-to-place, then slot index.
- Completed: all-invalid observations return a controlled no-op result with `reason: all_actions_invalid` for both random and greedy.
- Completed: updated `tests/test_protocol_contracts.py` to match the current active cup-like grasp contract (`top_down` + `handle_top_down`) instead of requiring the removed legacy `handle_grasp` side candidate.
- Verification: `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"` passed; `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed; `python -m unittest discover -s tests -p "test_protocol_contracts.py"` passed; modified Python files compile with `syntax ok`.
- Recommended next step: owner group `logic/fsm`; goal: implement `CupMugOrderingEpisodeRunner` in `rl/cup_mug_runner.py`; success criteria: a mocked 5-object episode completes after five valid actions, invalid actions never call the executor, low-level failures are logged, and all-invalid observations terminate cleanly; suggested verification: run `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"`.

## 2026-05-03 RL Cup/Mug Ordering Dry-Run Integration + Early Trace Visibility

- Completed: added `rl/cup_mug_runner.py` so cup-ordering episodes now execute through a pure, mockable runner with per-step `step_logs`, `selected_order`, reward breakdown, invalid-action handling, and all-invalid termination.
- Completed: added `rl/cup_mug_metrics.py` and extended `runtime/run_logger.py` so ordering runs can emit aggregate metrics JSON plus nested `rl_ordering` run-log payloads without breaking legacy log contracts.
- Completed: extended `rl/harness.py` and `scripts/eval_rl.py` so `cup_ordering` dry-run eval no longer requires a real checkpoint for `random` / `greedy`; it now writes both metrics and a trace report under `outputs/rl_cup_mug_ordering/reports/`.
- Completed: the early non-video visibility path is now in place. Example trace artifact `outputs/rl_cup_mug_ordering/reports/greedy-episodes-30-trace.json` shows per-step `selected_slot`, `object_id`, `grasp_strategy`, `place_zone`, `reward`, and success, so the user can inspect ordering behavior before simulator-rendered video is added.
- Completed: train dry-run sanity now writes `outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-sanity-step-5.ckpt.json` with `mode: cup_ordering`, `action_space: Discrete(5)`, `max_targets: 5`, and `sanity: true`.
- Verification: `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"` passed with 16 tests; `python -m unittest discover -s tests -p "test_protocol_contracts.py"` passed with 25 tests; `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 16 tests.
- Verification: `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 30 --dry-run` wrote `outputs/rl_cup_mug_ordering/metrics/random-episodes-30.json`; `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 30 --dry-run` wrote `outputs/rl_cup_mug_ordering/metrics/greedy-episodes-30.json` and `outputs/rl_cup_mug_ordering/reports/greedy-episodes-30-trace.json`.
- Current acceptance-ready artifacts: `outputs/rl_cup_mug_ordering/metrics/random-episodes-30.json`, `outputs/rl_cup_mug_ordering/metrics/greedy-episodes-30.json`, `outputs/rl_cup_mug_ordering/reports/greedy-episodes-30-trace.json`, and `outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-sanity-step-5.ckpt.json`.

## 2026-05-03 RL Cup/Mug Ordering RL Dry-Run + Render Path

- Completed: `policy=rl` dry-run eval now executes through the same runner/metrics path as random/greedy and writes `outputs/rl_cup_mug_ordering/metrics/rl-episodes-1.json` plus `outputs/rl_cup_mug_ordering/reports/rl-episodes-1-trace.json` instead of stopping at print-summary only.
- Completed: cup-ordering dry-run render now writes deterministic frame artifacts and wires them into the trace report as `frame_paths`, so the user can correlate `selected_slot` / `object_id` / `grasp_strategy` / `place_zone` with step-by-step visuals.
- Completed: cup-ordering dry-run `--save-video` now degrades gracefully. If `imageio` is installed, it writes `.mp4`; if not, the run still succeeds and the trace report records `video_skip_reason: imageio_not_installed` while preserving frames and metrics.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests after adding RL dry-run + render/video assertions.
- Verification: `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 1 --dry-run` wrote `outputs/rl_cup_mug_ordering/metrics/rl-episodes-1.json`.
- Verification: `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 1 --dry-run --render --save-video --video-path outputs/rl_cup_mug_ordering/videos/greedy-episodes-1-manual.mp4` completed successfully; `outputs/rl_cup_mug_ordering/reports/greedy-episodes-1-trace.json` now records `video_output_path` and `video_skip_reason: null`, and the mp4 file exists.
- Acceptance-ready visual artifacts now include JSON trace + PNG frames + mp4 output in the current environment.
- Simulator-backed render assessment: current `cup_ordering --render` still uses synthetic trace cards, but `RobocasaEnvWrapper` already exposes live RGB/depth access. The next real integration step is to prefer wrapper camera frames when a real RoboCasa-backed cup-ordering execution path is enabled, with synthetic frames as the fallback.
- Recommended next step: owner group `integration/merge`; goal: wire the current cup-ordering runner/eval path to a real RoboCasa environment wrapper so `--render` / `--save-video` can capture simulator camera frames for the same episode ids and step logs, instead of synthetic trace cards.

## 2026-05-03 RL Cup/Mug Ordering Real-Background Render Attempt

- Completed: implemented a real-background render attempt in `rl/harness.py`. During cup-ordering dry-run `--render`, the pipeline now tries to create a real `RobocasaEnvWrapper`, reset the `CupMugSorting` scene, render once, and use the resulting camera RGB frame as the background under the same step trace overlays.
- Completed: if real-frame capture fails, the trace now records `frame_source: synthetic_fallback` and a structured `real_render_error`, instead of silently pretending the output is real simulator imagery.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests after adding render-source assertions.
- Verification: debug trace `outputs/rl_cup_mug_ordering/reports/greedy-episodes-1-realbg-debug.json` confirms the current blocker is environment/runtime availability, not trace/video logic.
- Follow-up verification in the dedicated RoboCasa environment succeeded: `conda run -n robotic-robocasa-rl python -c "... _run_cup_ordering_dry_run(... report_stem='greedy-episodes-1-realbg-conda5' ...)"` wrote `outputs/rl_cup_mug_ordering/reports/greedy-episodes-1-realbg-conda5.json`, and that trace records `frame_source: real_robocasa_camera` with `real_render_error: null`.
- Current acceptance status: the dry-run visualization path now supports both real mp4 output and real RoboCasa camera backgrounds when executed inside the `robotic-robocasa-rl` environment. Outside that environment, trace reports still correctly fall back to synthetic frames with a structured error reason.

## 2026-05-03 RL Cup/Mug Ordering Final Phase-1 Closeout

- Completed: render artifacts now also emit per-episode sidecar JSON files (for example `outputs/rl_cup_mug_ordering/reports/greedy-episode-001.sidecar.json`) that summarize `policy`, `episode_id`, `selected_order`, `object_ids`, `has_handle`, `grasp_strategy`, `place_zone`, `success`, and failure fields alongside frame/video paths.
- Completed: `RL.md` has been lightly corrected so active Phase-1 wording matches the implemented contract: `MAX_TARGETS=5`, class-specific sink/opposite-counter placement, and no claim that RL already outperforms greedy.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests after sidecar support and `RL.md` guardrail cleanup.
- Verification: `python -c "from pathlib import Path; text=Path('RL.md').read_text(encoding='utf-8'); assert 'MAX_CUPS=4' not in text; assert 'max_cups=4' not in text; assert 'Discrete(4)' not in text; assert 'MAX_TARGETS=5' in text; print('rl doc checked')"` passed.
- Acceptance-ready artifact set: `outputs/rl_cup_mug_ordering/metrics/random-episodes-30.json`, `outputs/rl_cup_mug_ordering/metrics/greedy-episodes-30.json`, `outputs/rl_cup_mug_ordering/metrics/rl-episodes-1.json`, `outputs/rl_cup_mug_ordering/reports/greedy-episodes-30-trace.json`, `outputs/rl_cup_mug_ordering/reports/greedy-episodes-1-realbg-conda5.json`, `outputs/rl_cup_mug_ordering/reports/greedy-episode-001.sidecar.json`, and `outputs/rl_cup_mug_ordering/videos/greedy-episodes-1-manual.mp4`.
- Recommended next step: either (1) document a single canonical end-user reproduction command set for standard-B acceptance, or (2) move from dry-run ordering demonstration to a true online full-episode `policy -> vision -> grasp/place` continuous viewer demo in one MuJoCo session.

## 2026-05-03 Online Greedy + Vision + Grasp Demo Runner

- Completed: extracted the single-target live execution sequence into `arm/cup_mug_live_execution.py`, so the existing single-target Cup/Mug live path can be reused by a continuous online ordering demo without duplicating reach / close / lift / place orchestration.
- Completed: added `rl/cup_mug_live_mapping.py` to convert current live `targets + assignments + sorting_metadata` into the same 5-slot observation semantics used by dry-run ordering, plus `slot_index -> current live target/assignment` resolution.
- Completed: added `scripts/demo_online_cup_mug_ordering.py`, a first online MuJoCo viewer runner that keeps one `CupMugSorting` session alive, re-senses the scene each step, chooses the next slot with `greedy`, resolves the current live target, and runs the real single-target execution helper.
- Verification: syntax checks passed for `arm/cup_mug_live_execution.py`, `rl/cup_mug_live_mapping.py`, and `scripts/demo_online_cup_mug_ordering.py`; `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"` passed with 18 tests; `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests.
- Verification: running `C:\Users\BeautifulLee\miniconda3\envs\robotic-robocasa-rl\python.exe scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy greedy --layout 1 --style 1 --keep-open-sec 0.2 --save-trace --trace-path outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json` succeeded and wrote `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json`.
- Current online demo status: the runner reaches a real continuous episode in one viewer session and records `online_demo_step` / `online_demo_summary`; the observed run completed multiple picks (`mug_1`, `mug_2`, `cup_1`) before stopping on a later `reach` failure, which is now captured as structured trace data rather than an opaque terminal-only failure.

## 2026-05-03 Online Demo Controlled Retry Upgrade

- Completed: upgraded the online demo failure policy so a failed selected slot no longer causes an immediate episode stop. The runner now performs one controlled retry on the same slot after re-sensing the scene and rebuilding the live slot mapping.
- Completed: added a place-only retry path in `arm/cup_mug_live_execution.py`; if `place` fails after a successful grasp/lift, the system recalculates a fresh `place_plan` and retries `execute_place()` once before giving up.
- Completed: `scripts/demo_online_cup_mug_ordering.py` now records per-step `attempts`, retry refresh validity, and retry-related failure metadata in the trace artifact.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` remained green with 19 tests; syntax checks passed for the updated online runner and live execution helper.
- Verification: running `C:\Users\BeautifulLee\miniconda3\envs\robotic-robocasa-rl\python.exe scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy greedy --layout 1 --style 1 --keep-open-sec 0.2 --save-trace --trace-path outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json --retry-on-failure-once` succeeded and wrote a trace where `finished` includes `mug_1`, `mug_2`, and `cup_1`.
- Current online demo status: completion depth improved from early reach/place termination to three successful objects in one continuous viewer session; the current terminal condition is now `all_actions_invalid`, meaning the next bottleneck is live re-detection/slot validity for the remaining objects rather than immediate execution collapse.
- Recommended next step: investigate why remaining visible cups collapse into invalid slots after several successful picks, then adjust post-place re-sensing / visibility handling so the 5-slot observation preserves valid actions deeper into the episode.

## 2026-05-03 Online Demo Mapping + Place Retry Follow-up

- Completed: `vision/sorting_policy.assign_sim_metadata_to_targets()` now supports a controlled second-pass nearest-neighbor fallback for still-unmatched visual targets, reducing late-episode loss of `sim_object_name` after objects move.
- Completed: `arm/cup_mug_live_execution.py` now retries `place` once by recomputing `place_plan` from refreshed assignment pose instead of forcing a full re-grasp immediately.
- Verification: `python -m unittest discover -s tests -p "test_sorting_policy.py"` passed with 6 tests; `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests after the mapping/online retry changes.
- Verification: rerunning the online greedy demo with `--retry-on-failure-once` still succeeds as a process and writes `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json`; the latest trace shows `retry_refresh_slot_valid=true` on the retried slot, so live mapping stayed intact.
- Current online demo status: the dominant bottleneck has shifted again. Slot validity no longer collapses first; the remaining failure is a later `handle_top_down` reach on `mug_2` second attempt, with `reachability_diagnosis.likely_cause = needs_action_space_or_reachability_probe`.

## 2026-05-03 Online Demo Handle-Target Conservative Defaults

- Completed: the online runner now automatically applies a more conservative execution profile for `has_handle=true` slots, enabling handle-target-specific defaults such as `enable_base_torso_preposition`, `auto_preposition_retry`, and `retry_place_once` without changing cup defaults.
- Verification: rerunning the online greedy demo shows these defaults are active in trace metadata (`base_torso_preposition.enabled=true` on handle-target attempts).
- Current limitation: the same trace also shows `effective_should_move=false`, so the conservative policy is enabled but does not yet translate into actual base motion in the current run. The dominant failure remains a later `handle_top_down` reach issue rather than slot-mapping collapse.
- Recommended next step: make calibrated base action mapping the default for handle-target online attempts so the preposition stage can produce meaningful movement instead of remaining a no-op.

## 2026-05-03 Online Demo Handle Base-Mapping Success

- Completed: for handle targets, the online runner now auto-calibrates base action mapping when needed and applies a more aggressive but still bounded preposition policy (`desired_xy_standoff=0`, `base_xy_deadband=0`, lower `base_action_limit`) before executing `handle_top_down`.
- Verification: rerunning `scripts/demo_online_cup_mug_ordering.py` in the RoboCasa environment produced `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json` with `selected_order = [0, 0, 3, 1, 2]` and `finished` containing all five target objects (`mug_2`, `mug_1`, `cup_3`, `cup_1`, `cup_2`).
- Verification: handle-path trace metadata now shows `base_mapping_used=true`, `effective_should_move=true`, and `xy_distance_improved=true`, confirming that preposition is no longer a no-op for handle targets.
- Current online demo status: the greedy + vision + real grasp/place runner can now complete a full 5-object CupMugSorting episode in one continuous MuJoCo viewer session under the verified RoboCasa environment.
- Recommended next step: if desired, promote the same online execution path from `policy=greedy` to `policy=rl` (keeping the same live slot mapping and execution helper) once the learned policy is ready to replace the current RL sanity stub.

## 2026-05-03 Sink Split Placement Update

- Completed: updated the placement semantics from `mug -> sink / cup -> opposite_counter` to `mug -> right_sink / cup -> left_sink` so both categories place into distinct sink basins rather than splitting sink/counter.
- Completed: `planner/sorting_zones.py`, `configs/tasks/cup_mug_sorting.yaml`, and `configs/rl/cup_mug_ordering_robocasa.yaml` now align on the left/right sink split contract.
- Verification target: `tests/test_sorting_zones.py` asserts handled mugs place into `right_sink` and plain cups place into `left_sink` when sink anchors are available.

## 2026-05-03 Sink Split Online Validation Result

- Verification: after syncing `rl/contracts.py` and `tests/test_rl_contracts.py` to the new sink-split contract, `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests and `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed with 4 tests.
- Online result: rerunning the online greedy demo with the new `mug -> right_sink / cup -> left_sink` placement contract succeeded as a process, but the latest `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json` only completed the two mug objects before terminating with a later `reach` failure.
- Interpretation: the sink-split policy is implemented correctly and contract-consistent, but under the current online execution stability it is less robust than the previous placement policy that had already reached a full 5/5 episode.

## 2026-05-03 RL Cup/Mug Online Checkpoint Interface Reservation + Failure Classification

- Completed: confirmed the current online ordering path had already finished most stage-1 work in code: live 5-slot mapping, retry-on-failure with re-sense/remap, later-step handle-target conservative execution defaults, base-mapping auto-calibration, and JSON trace persistence.
- Completed: added a reserved `policies.checkpoint_interface` contract to `configs/rl/cup_mug_ordering_robocasa.yaml` and `rl/contracts.py`, so cup-ordering now explicitly records its checkpoint artifact kind, schema version, and the current `supports_inference=false` capability boundary instead of implying real learned inference already exists.
- Completed: `rl/harness.py` now exposes shared cup-ordering checkpoint inspection / RL-policy selection helpers. The current dry-run sanity artifact is inspectable as a checkpoint-shaped artifact, but eval/online checkpoint mode now fail with a controlled capability error instead of silently falling back or using a raw `NotImplementedError`.
- Completed: `scripts/demo_online_cup_mug_ordering.py` no longer throws a raw `NotImplementedError` for `--policy rl --rl-policy-mode checkpoint`; it now reports structured `checkpoint_interface`, `failure_reason`, and `failure_category` fields, and can persist that status to trace JSON.
- Completed: online RL trace summaries now include an explicit `failure_category` in addition to raw `failure_phase` / `failure_reason`, so later analysis can distinguish policy-no-valid-action, slot-resolution, candidate-build/threshold, execution-phase, and checkpoint-interface failures.
- Verification: `tests/test_rl_contracts.py` now covers config summary, reserved checkpoint artifact metadata, checkpoint inspection, stub RL action continuity, and controlled eval failure for checkpoint mode.
- Recommended next step: owner group `logic/fsm`; goal: implement a true cup-ordering checkpoint inference backend that consumes the existing 5-slot observation contract and replaces the current sanity stub without changing online trace schema or low-level execution helpers. Success criteria: `--policy rl --rl-policy-mode checkpoint --rl-checkpoint <artifact>` selects real actions from model outputs, and dry-run/online traces clearly report checkpoint-backed policy metadata. Suggested verification: add focused tests for observation flattening + model output decoding, then rerun `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --checkpoint <artifact> --episodes 1 --dry-run` plus the online viewer command in the RoboCasa environment.
- Recommended decision point: if the highest priority is a stable end-to-end online demo, the safer fallback is to keep mugs in sink but return cups to a reachable right-side tabletop placement with non-stacking offsets. If the highest priority is specifically demonstrating left/right sink semantics, we should expect more execution tuning work before the online viewer demo becomes consistently reliable again.

## 2026-05-03 Placement Fallback Back To Stable Counter Policy

- Completed: reverted the cup placement target from `left_sink` back to a more stable right-side counter placement, while keeping mugs in the sink.
- Completed: the counter placement path is now explicitly treated as a spread tabletop target (`placement_rule: spread_counter_edge`) rather than a single shared stacking point.
- Completed: `planner/sorting_zones.py`, `configs/tasks/cup_mug_sorting.yaml`, `configs/rl/cup_mug_ordering_robocasa.yaml`, `rl/contracts.py`, and `tests/test_rl_contracts.py` were synchronized to the restored stable policy (`mug -> sink`, `cup -> right_counter`).
- Recommended next step: re-run the online greedy demo and compare whether the full-episode completion returns to the earlier 5/5 behavior under the restored stable placement contract.

## 2026-05-03 Placement Fallback Online Verification

- Verification: after restoring the more stable placement policy, `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests and `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed with 4 tests.
- Verification: rerunning the online greedy demo with the restored policy still succeeded as a process and wrote `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json`, but this specific run terminated on a later `reach` failure after completing two mug objects.
- Interpretation: reverting cups from `left_sink` to `right_counter` is still the safer design choice for Phase 1, but the online completion rate is not determined by placement policy alone; real candidate/execution variability remains a dominant factor.
- Recommended next step: keep the stable placement policy (`mug -> sink`, `cup -> right_counter`) and focus the next debugging pass on grasp candidate quality / reachability robustness rather than reopening the placement design debate.

## 2026-05-03 Safe Retreat + Sink Spacing Follow-up

- Completed: `arm/robocasa_primitives.execute_place()` now adds a `safe_retreat` phase after the initial retract, with `safe_retreat_target` recorded in the place summary so post-place exits can be inspected explicitly.
- Completed: `planner/sorting_zones.py` now supports simple sink spacing for handled mugs (`placement_rule: spaced_sink_basin`) using small per-mug y-offsets instead of sending all handled mugs to the exact same basin point.
- Verification: `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed with 5 tests and `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 19 tests after these changes.
- Online result: the latest online greedy run still terminated on a later `reach` failure before sink crowding became the primary issue, so the new sink-spacing rule did not yet get a fair chance to improve end-state behavior in that run.
- Current interpretation: these two mitigations are now in place, but the dominant blocker remains later-step candidate / reach robustness rather than placement-exit geometry alone.

## 2026-05-03 Online Candidate Identity Preservation Follow-up

- Completed: the online execution candidate fallback path now preserves `sim_object_name` identity when possible, preferring a fallback target for the exact intended sim object instead of degrading to a generic label-based cup/mug fallback.
- Verification: rerunning the online greedy demo with the fix still succeeded as a process and wrote `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json`.
- Observed effect: the inspected trace no longer shows the earlier obvious semantic corruption where a handle target could collapse into a generic `cup/top_down` fallback; the latest run completed `mug_1`, `mug_2`, and `cup_1` before stopping on a later `reach` failure.
- Current status: identity preservation is improved, but stable 5/5 completion is still blocked by later reachability/candidate variability rather than placement semantics alone.
- Recommended next step: focus the next pass on improving later-step candidate quality / reach robustness (especially after multiple successful object moves) rather than broadening fallback semantics further.

## 2026-05-03 Online Greedy Baseline 5/5 Restored

- Completed: the online candidate builder now prefers the current frame's fresh assignment for the same `sim_object_name` instead of over-trusting stale assignment state, reducing later-step semantic drift.
- Verification: rerunning the online greedy demo after this fix produced `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace.json` with `selected_order = [0, 1, 2, 3, 1]` and `finished` containing all five target objects (`mug_1`, `mug_2`, `cup_1`, `cup_3`, `cup_2`).
- Verification: the same trace records `retry_counts = {}` and `failure_reason = null`, showing that the latest online greedy baseline run finished cleanly without residual failure state.
- Current baseline status: the greedy + vision + real grasp/place online runner is now stable enough to serve as the reference baseline for the next phase, where the same live slot mapping and execution helper can be reused for `policy=rl` online evaluation.
- Recommended next step: start promoting `policy=rl` onto the same online execution path while keeping greedy as the comparison baseline.

## 2026-05-03 Online RL Execution Entry Enabled

- Completed: `scripts/demo_online_cup_mug_ordering.py` now accepts `--policy rl` and routes it through the same live slot mapping, MuJoCo viewer session, and real grasp/place execution chain as `greedy`.
- Completed: `LHYstart.md` now includes both the recommended online greedy command and the current online RL sanity command.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` remained green with 19 tests after adding the online RL entry.
- Current RL status: the online RL path is connected at the control-flow level, but it still uses the existing sanity/stub ordering policy and needs its own stability pass before it should be presented as a polished online demo.
- Recommended next step: keep `greedy` as the online gold baseline and perform one dedicated stabilization pass for `policy=rl` on the same execution chain before comparing behaviors.

## 2026-05-03 Online RL Quality-Biased Stub Verification

- Completed: the current RL sanity selector in `rl/harness.py` now uses a quality-biased stub ranking over valid actions instead of purely uniform random selection.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` remained green with 19 tests after the RL sanity selector update.
- Verification: running the online RL command in the RoboCasa environment enters the same online execution path and viewer flow as the greedy baseline.
- Current RL blocker: the latest run failed at `candidate_build` for a later handle target because the online handle-candidate gate rejected a candidate with score `0.2871`, slightly below the current `0.30` threshold. This means the remaining RL online issue is now a threshold/policy interaction, not a missing integration path.
- Recommended next step: perform one focused RL-specific stabilization pass for handle-target candidate acceptance (or policy-side slot preference) before using `policy=rl` as a serious online comparison against the restored greedy 5/5 baseline.

## 2026-05-03 Online RL Later-Step Stabilization In Progress

- Completed: the RL sanity selector now uses a quality-biased stub ranking over valid actions instead of purely random sampling.
- Completed: the online RL path is still using the same live slot mapping and execution helper as greedy, preserving apples-to-apples execution semantics.
- Current status: later-step RL stabilization is still incomplete. The dominant issue remains handle-target candidate acceptance / consistency after one or more successful steps, rather than wiring or viewer integration.
- Additional note: the latest code path indicates one direct implementation issue to fix next (`step_index`-dependent threshold logic needs to be passed through `_build_live_execution_candidate()` cleanly) before drawing stronger conclusions from the newest RL online attempts.
- Recommended next step: fix the `step_index` threshold plumbing bug, then rerun the online RL demo and continue the RL-only later-step handle tolerance pass while keeping greedy as the gold baseline.

## 2026-05-03 Online RL Later-Step Tolerance Pass

- Completed: the RL sanity selector now uses a `quality_biased_stub` ordering policy rather than pure random valid-slot sampling, and the online candidate gate now records RL-specific threshold metadata in trace output.
- Completed: the later-step `step_index` threshold plumbing bug was fixed, so RL-specific handle thresholds are now applied intentionally rather than accidentally.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` remained green with 19 tests.
- Verification: rerunning the online RL command still enters the full viewer / slot mapping / execution chain, but the latest run continues to stop after completing one handled object, with `failure_reason = candidate_build` on a later handle target.
- Current RL online status: the remaining work is no longer integration. It is now a narrow RL-specific threshold / candidate-acceptance tuning problem over an already working online control path.
- Recommended next step: if we continue on the RL line, the next pass should directly experiment with RL-only handle thresholds / fallback acceptance policy against the preserved greedy 5/5 baseline rather than reopening the broader online architecture.

## 2026-05-03 Online RL Checkpoint Inference Path

- Completed: reviewed `.sisyphus/plans/rl-cup-mug-final-training.md` and `.sisyphus/plans/online-greedy-vision-grasp-demo.md`; Momus marked both plans `[OKAY]`. Current code already exceeds the original greedy-only online plan, so the next useful slice was narrowed to replacing the RL sanity selector path with a real checkpoint inference seam.
- Completed: `rl/harness.py` now exposes `flatten_cup_ordering_observation()` for the fixed 5-slot observation contract. The vector is stable length 81: five slots × 14 numeric features, six global features, and five action-mask features.
- Completed: `select_cup_ordering_rl_action()` now supports checkpoint mode by loading `.zip` PPO artifacts through `stable_baselines3.PPO.load()` and calling `model.predict(flattened_observation, deterministic=True)`. The returned discrete action is decoded to a slot index and trace debug records `selection_mode=checkpoint_inference`, `predicted_action`, `predicted_action_valid`, `observation_vector_length`, and checkpoint metadata.
- Completed: `configs/rl/cup_mug_ordering_robocasa.yaml` now marks the checkpoint interface as `supports_inference: true`; dry-run sanity JSON artifacts remain explicitly non-inference-capable via `supports_inference=false` and `unsupported_reason=cup_ordering_sanity_artifact_has_no_model_weights`.
- Completed: `scripts/eval_rl.py` now reports controlled CLI errors for missing RL runtime / unsupported checkpoint problems instead of printing a raw traceback.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 25 tests, including fake-PPO coverage for checkpoint loading and model prediction without requiring local `stable_baselines3`.
- Verification: `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 1 --dry-run` still writes RL sanity metrics through the existing stub path when no checkpoint is supplied.
- Local limitation: this environment does not have `stable_baselines3` installed and only contains sanity JSON artifacts under `outputs/rl_cup_mug_ordering/checkpoints/`; therefore real `.zip` checkpoint inference is code-tested with a fake PPO and must still be exercised in the RoboCasa/SB3 environment with a real trained checkpoint.
- Recommended next step: produce or provide a real cup-ordering `.zip` PPO checkpoint whose observation input matches the 81-feature vector, then run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 1 --dry-run --checkpoint <checkpoint.zip>` and the online command `python scripts/demo_online_cup_mug_ordering.py --policy rl --rl-policy-mode checkpoint --rl-checkpoint <checkpoint.zip> --layout 1 --style 1 --save-trace --keep-open-sec 1` inside the `robotic-robocasa-rl` / SB3-capable environment.

## 2026-05-03 Online RL Stub Handle-First Stabilization

- Completed: inspected the current `outputs/rl_cup_mug_ordering/reports/online-rl-demo-trace.json` against the successful greedy trace. The RL sanity/stub path completed `mug_1`, then selected `cup_3` and stopped on `failure_reason=reach`; the successful greedy reference cleared both handled mugs before moving to plain cups.
- Completed: updated the RL sanity selector in `rl/harness.py` from pure quality-biased ranking to `handle_first_quality_biased_stub`. It still consumes the same valid-action mask and quality terms, but now ranks valid `has_handle=true` slots before plain cups; after handle slots are finished/masked, it returns to quality ranking among plain cups.
- Rationale: this keeps RL within the same high-level ordering-only contract while matching the empirically stable online execution shape: clear handle targets first, then execute top-down cup placements. It does not change greedy, live mapping, grasp strategy, placement zones, or low-level execution helpers.
- Verification: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 27 tests, including explicit coverage that RL stub prioritizes handle targets over higher-scoring plain cups and falls back to quality ranking after handle targets are finished.
- Suggested online verification: rerun `python scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy rl --layout 1 --style 1 --keep-open-sec 0.2 --save-trace --trace-path outputs/rl_cup_mug_ordering/reports/online-rl-demo-trace.json --retry-on-failure-once` inside the RoboCasa environment. Success criteria: `finished` contains all five target objects and `failure_reason=null`; trace `policy_debug.selection_mode` should show `handle_first_quality_biased_stub`.

## 2026-05-03 Online RL Stub 5/5 Stability Pass

- Completed: after the handle-first RL stub change, a real RoboCasa online rerun exposed a retry-time identity drift (`mug_1` retry remapped to `mug_2`) and a separate `_build_live_execution_candidate()` bug where `fresh_assignments` could be referenced before assignment. Both were fixed: retry refresh now prefers the original `sim_object_name` when it is still valid, and `fresh_assignments` is always initialized before being referenced.
- Completed: `rl/cup_mug_live_mapping.py` now provides `resolve_live_target_with_identity_preference(...)`, and `scripts/demo_online_cup_mug_ordering.py` uses it on retry refresh so the same decision step keeps targeting the same object identity when possible.
- Verification: local focused regression remained green after the retry/identity fix: `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"` passed with 20 tests, and `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 27 tests.
- Real online result: running `C:\Users\BeautifulLee\miniconda3\envs\robotic-robocasa-rl\python.exe scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy rl --layout 1 --style 1 --keep-open-sec 0.2 --save-trace --trace-path outputs/rl_cup_mug_ordering/reports/online-rl-demo-trace-stability-pass-3.json --retry-on-failure-once` completed a full 5/5 episode.
- Evidence from `outputs/rl_cup_mug_ordering/reports/online-rl-demo-trace-stability-pass-3.json`: `selected_order = [1, 0, 1, 0, 0]`, `finished` contains all five target objects (`mug_2`, `mug_1`, `cup_2`, `cup_1`, `cup_3`), `retry_counts = {}`, `failure_reason = null`, and step-level `policy_debug.selection_mode = handle_first_quality_biased_stub`.
- Current status: the sanity/stub RL path now reaches the same live slot mapping + real grasp/place chain as greedy and has demonstrated a full 5/5 online completion in the validated RoboCasa environment. This is a stability milestone for the stub ordering policy, not evidence of learned-policy superiority.
- Recommended next step: preserve the new 5/5 trace as the reference RL-sanity online artifact, then either (1) repeat the same command across a small seed/layout set to measure robustness variance, or (2) switch back to the checkpoint path and start validating a real learned policy against the same live execution contract.

## 2026-05-03 Greedy Placement / Retreat / Handle-Threshold Follow-up

- Completed: cup counter placement targets are no longer a single shared point. `planner/sorting_zones.py` now offsets same-category cup targets along local counter `y` using the object suffix (`cup_1`, `cup_2`, `cup_3`) so repeated plain-cup placements are less likely to stack directly on top of one another. `tests/test_sorting_zones.py` now asserts distinct counter targets for multiple cups.
- Completed: `arm/robocasa_primitives.execute_place()` now uses a more conservative post-place retreat: `safe_retreat_height` was increased from `0.22` to `0.30`, and `retract_target` no longer drops below the pre-place end-effector height.
- Real result: the first greedy retreat-focused rerun (`online-greedy-demo-trace-retreat-pass.json`) still failed before any placement-side benefit could be assessed, because the run terminated immediately on a handle candidate threshold miss (`candidate_score=0.2882 < 0.30`). This weakens the hypothesis that retreat height was the primary current blocker.
- Completed: a greedy-only narrow handle-candidate margin accept was added for strong geometry cases (`0.25 <= score < 0.30` with sufficient `handle_point_count`, `protrusion_quality`, and bounded `gripper_width`), without changing RL behavior or the nominal `0.30` threshold for ordinary cases.
- Real result: the second threshold-focused rerun (`online-greedy-demo-trace-threshold-pass-2.json`) progressed further than the immediate threshold-fail case and successfully completed `mug_1`, but still terminated later on `mug_2` with `failure_reason=reach` and `retry_counts={"mug_2": 2}`. This indicates the threshold softening helps the earliest gate, but later-step handle execution remains unstable.
- Current interpretation: placement spread and higher safe-retreat are sensible mitigations, but the dominant greedy blocker is still later-step handle robustness rather than cup stacking alone. Threshold softening reduced one early failure mode, yet did not restore stable 5/5 completion by itself.
- Recommended next step: if greedy visual stability remains the priority, focus the next pass on later-step handle execution robustness (for example, conservative handle candidate acceptance conditioned on step index / retry state, or tighter post-success re-sense sanity checks) before revisiting placement geometry again.

## 2026-05-03 Greedy Later-Step Handle Stability Pass

- Completed: enabled one controlled base-preposition retry path for `handle_top_down` by expanding `arm/reach_retry.should_retry_with_preposition()` from `("top_down",)` to `("top_down", "handle_top_down")`. This does not change policy ordering; it only lets later-step handle reaches use the same single recovery mechanism already available to top-down reaches.
- Real result: `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace-later-step-pass.json` progressed further than the earlier threshold-only pass. The run completed `mug_1` successfully, then advanced to `mug_2` and reached the place phase before failing, instead of terminating immediately on the first handle candidate gate.
- Evidence from the later-step trace: `mug_2` no longer died at the first reach gate. The first `mug_2` attempt reached/closed/lifted successfully but failed at `failure_phase=place`; the second attempt then fell back to a new `candidate_build` failure with `candidate_score=0.2893 < 0.30`. This indicates measurable progress in handle reach robustness, but not a full greedy 5/5 recovery yet.
- Current interpretation: the dominant greedy blocker has shifted again. Early handle gating is less brittle than before, and later-step handle reach is more resilient, but the remaining instability now spans both post-grasp placement and retry-time candidate consistency for handled targets.
- Recommended next step: if we continue on greedy, the most targeted follow-up is no longer generic retreat height or cup spread; it is handled-object place stability plus retry-time handle candidate retention for the second mug.

## 2026-05-03 Greedy Retreat Ceiling Fix + 5/5 Recovery

- Completed: investigated the new greedy failure mode where the arm appeared to rise into upper cabinet geometry after placing a handled mug. The direct code path was `arm/robocasa_primitives.execute_place()`, where `safe_retreat_target.z` had been raised to `max(start_pos[2], place_target.z + 0.30)` with no ceiling-aware clamp.
- Evidence: `online-greedy-demo-trace-later-step-pass.json` showed a handled `place` failure whose `history_tail` stayed in `phase=safe_retreat` while trying to reach `safe_retreat_target.z ≈ 1.317`, but the end-effector stalled near `z ≈ 1.243` with residual error around `0.075`, matching the user's observation that the arm was effectively blocked by overhead geometry rather than failing at grasp or reach.
- Completed: added a conservative ceiling cap in `arm/robocasa_primitives.execute_place()` so `safe_retreat_target.z` no longer exceeds `1.24`. This preserves the vertical retreat logic but prevents the arm from blindly pushing into likely upper-cabinet space.
- Verification: local focused regression remained green after the retreat-cap change: `python -m unittest discover -s tests -p "test_rl_contracts.py"` passed with 27 tests and `python -m unittest discover -s tests -p "test_sorting_zones.py"` passed with 6 tests.
- Real online result: rerunning greedy with `C:\Users\BeautifulLee\miniconda3\envs\robotic-robocasa-rl\python.exe scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy greedy --layout 1 --style 1 --keep-open-sec 0.2 --save-trace --trace-path outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace-retreat-cap-pass.json --retry-on-failure-once` completed a full 5/5 episode.
- Evidence from `outputs/rl_cup_mug_ordering/reports/online-greedy-demo-trace-retreat-cap-pass.json`: `selected_order = [1, 0, 0, 2, 1]`, `finished` contains all five target objects (`mug_2`, `mug_1`, `cup_1`, `cup_3`, `cup_2`), `retry_counts = {}`, `failure_reason = null`, and `last_success = true`.
- Current greedy status: with cup counter spread, handle threshold softening, handle later-step retry, and the retreat ceiling cap together, the greedy online visual path has recovered a real 5/5 completion in the validated RoboCasa environment.
- Recommended next step: preserve `online-greedy-demo-trace-retreat-cap-pass.json` as the current greedy reference artifact, then repeat the same command across a few randomized runs to see whether the retreat ceiling fix improved robustness or just repaired one layout/scene interaction.
