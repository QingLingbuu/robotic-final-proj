# 项目任务总结

> 维护约束：凡是影响项目现状、已完成改动、下一步优先级或剩余任务清单的开发工作，必须同步更新本文件 `docs/task.md`，避免实现进度与任务文档脱节。

## 一、当前状态

当前仓库已经从“混合实验目录”收口成较清晰的工程结构：

- 主业务代码位于 `arm/`、`fsm/`、`vision/`、`rl/`
- 运行时支撑统一收口到 `runtime/`
- 训练、评估、setup、validate 脚本统一归到 `scripts/`
- 外部源码统一 vendoring 到 `third_party/robocasa` 与 `third_party/robosuite`
- RL 输出路径统一固定到 `outputs/rl/`

项目当前仍以 `robosuite` 作为稳定联调基线，同时保留可运行的 `RoboCasa` 单臂 RL smoke 路径。

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
- 新增和整理文档：
  - [Getting Started](/D:/Code/MyRepositories/robotic-final-proj/docs/getting-started.md)
  - [Docs Index](/D:/Code/MyRepositories/robotic-final-proj/docs/index.md)
  - [RoboCasa Migration Architecture](/D:/Code/MyRepositories/robotic-final-proj/docs/architecture/robocasa-migration.md)
  - [2026-04 History Summary](/D:/Code/MyRepositories/robotic-final-proj/docs/archive/2026-04-history.md)

## 三、当前主要问题

### 1. `main.py` 仍然偏重

虽然根目录和支撑目录已经明显更干净，但 [main.py](/D:/Code/MyRepositories/robotic-final-proj/main.py) 仍然承担了较重的 orchestration 责任，还不是一个足够薄的主入口。

### 2. RoboCasa 路径仍有运行时警告

`gymnasium` observation-space warning 仍未收敛。当前 smoke path 已可运行，但还没有把这类 runtime mismatch 彻底收口。

### 3. 测试环境仍受 Windows 权限影响

当前最典型的问题不是 contract 本身，而是：

- `multiprocessing.Queue()` 在部分测试环境下权限拒绝
- 测试里创建 `outputs/` 目录时权限拒绝

这意味着仓库结构已经更整齐，但测试环境还不够干净。

### 4. `third_party/robocasa` 体积较大

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

- owner group: `infra`
- goal: 继续把仓库从“能用”推进到“结构稳定”

### 优先级 1

继续收薄 [main.py](/D:/Code/MyRepositories/robotic-final-proj/main.py)，把 orchestration 再拆一层，让主入口更像主入口，而不是集成脚本。

### 优先级 2

清理测试环境权限问题，优先解决：

- `multiprocessing.Queue()` 权限拒绝
- 测试创建 `outputs/` 目录权限拒绝

### 优先级 3

决定 `third_party/robocasa` 的长期策略：

- 保留完整 vendored 资产
- 或改为“源码进仓、部分资产外置下载”

## 六、建议验证

- `python scripts/setup/print_robocasa_rl_setup.py`
- `python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary`
- `python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary`
- `python -m unittest discover -s tests -p "test_dependency_baseline.py"`
- `rg -n "runtime/|third_party/|scripts/setup|scripts/validate" docs scripts tests arm main.py rl vision fsm -S`

