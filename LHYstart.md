# LHYstart

这份文档是给当前这轮 **robosuite 抓取诊断 / 对照脚本** 准备的快速上手说明。  
如果你只想知道“环境怎么启动、命令怎么跑、该先看哪个脚本”，优先看这里。

> 说明：更完整的仓库环境策略和项目级说明仍以 `docs/getting-started.md` 为准；本文件偏向**当前机器 / 当前这轮调试**里实际使用过的命令备忘，不替代项目正式环境策略说明。

---

## 1. 当前常用环境

本轮在当前机器上，实际用于运行以下脚本的环境是：

```powershell
conda activate robotic-robocasa-rl
```

这些脚本包括：

- `scripts/demo_vision_grasp.py`
- `scripts/demo_GT_grasp.py`
- `scripts/checkVision.py`

如果当前机器上还没创建这个环境，可在项目根目录执行：

```powershell
conda create -n robotic-robocasa-rl python=3.10 pip
conda activate robotic-robocasa-rl
pip install -r requirements.txt
```

建议先做一次依赖 smoke check：

```powershell
python scripts/setup/check_robocasa_rl_deps.py
```

---

## 2. 最常用的启动方式

> 如果你想看项目级“应该如何分环境”的正式说明，请回到 `docs/getting-started.md`。本节只记录这轮本地调试时最常跑的命令。

### 2.1 Vision 抓取 demo

```powershell
python scripts/demo_vision_grasp.py --scenario cube --render
python scripts/demo_vision_grasp.py --scenario can --render
python scripts/demo_vision_grasp.py --scenario milk --render
python scripts/demo_vision_grasp.py --scenario bread --render
python scripts/demo_vision_grasp.py --scenario cereal --render
```

说明：

- 这是 **vision-driven** 路径
- 会走：`GroundingDINO -> detected_objects -> FSM -> candidate -> execute_grasp`
- 当前 `cube` 场景已经改回 **单臂 Lift**，不再使用之前那个“双臂环境里的单臂抓取”配置

如果只想跑默认场景：

```powershell
python scripts/demo_vision_grasp.py
```

### 2.2 GT 抓取 demo

```powershell
python scripts/demo_GT_grasp.py --scenario cube
python scripts/demo_GT_grasp.py --scenario can
python scripts/demo_GT_grasp.py --scenario milk
python scripts/demo_GT_grasp.py --scenario bread
python scripts/demo_GT_grasp.py --scenario cereal
```

说明：

- 这是 **ground-truth-driven** 路径
- 结构尽量贴近 `demo_vision_grasp.py`
- 主要区别是抓取目标直接来自 robosuite GT，而不是视觉检测结果
- 用来判断问题更偏向 **vision** 还是 **execution**

如需渲染：

```powershell
python scripts/demo_GT_grasp.py --scenario cube --render
```

### 2.3 抓取后放置测试

vision demo：

```powershell
python scripts/demo_vision_grasp.py --scenario can --place-test
```

GT demo：

```powershell
python scripts/demo_GT_grasp.py --scenario can --place-test
```

---

## 3. `checkVision.py` 的常用用法

### 3.1 单场景完整诊断

```powershell
python scripts/checkVision.py --scenario cube
python scripts/checkVision.py --scenario can
```

它会输出：

- `vision target`
- `selected candidate`
- `GT center / top surface`
- `planned grasp / pre_grasp / lift`
- `eef_before_close`
- 关键 delta（特别是 z 方向）

### 3.2 只看规划/几何，不执行抓取

```powershell
python scripts/checkVision.py --scenario cube --skip-execution
```

### 3.3 JSON 摘要

```powershell
python scripts/checkVision.py --scenario all --summary
```

### 3.4 表格摘要

```powershell
python scripts/checkVision.py --scenario all --summary-table
```

当前表格会重点显示：

- `gt_xyz`
- `vision_xyz`
- `cand_z-gt_top`
- `final_z-gt_top`
- `eef_z-plan_z`
- `eef_z-gt_top`
- `grasp_verified`
- `error`

---

## 4. 当前这轮诊断的已知结论

### 4.1 执行侧

当前单臂抓取里有一个重要事实：

- `robot0_eef_pos` 更像 **EEF/TCP 参考点**
- 它不等于真实指尖接触面

因此当前引入了：

```text
SINGLE_GRASP_CONTACT_Z_OFFSET = 0.028
```

这更适合被理解为：

> EEF 参考点到实际接触面的几何补偿

而不是“所有对象都通用的抓取误差常数”。

### 4.2 cube vs can

目前已经观察到：

- `cube` 在修正为单臂 Lift 并加入接触补偿后，GT 路径和 `checkVision` 路径都能成功抓起
- `can` 与 `cube` 共享近似的 EEF 参考点偏移量，但这并不意味着所有对象都可以靠一个固定 z offset 解释
- 更核心的问题是：**不同对象的 grasp target 语义并不相同**

### 4.3 当前剩余主要问题

`cube` 的 vision demo 仍可能因为 detector 给出 **超大背景框**，导致：

- `candidate.pos` 飘到明显不合理的位置
- `gripper_width` 变成异常大值

这时主问题已经不再是执行层，而是：

> vision 候选缺乏足够的几何合理性过滤

---

## 5. 遇到问题时优先看什么

### 如果脚本启动失败

先确认环境：

```powershell
conda activate robotic-robocasa-rl
python scripts/setup/check_robocasa_rl_deps.py
```

### 如果 `demo_vision_grasp.py` 抓不到

优先跑：

```powershell
python scripts/checkVision.py --scenario cube
```

看：

- `candidate_z_minus_gt_top_surface`
- `eef_before_close_z_minus_planned_grasp_z`
- `grasp_verified`
- `vision_xyz` / `gt_xyz`

### 如果怀疑是 vision 还是 execution

对照跑：

```powershell
python scripts/demo_vision_grasp.py --scenario cube
python scripts/demo_GT_grasp.py --scenario cube
```

判断原则：

- **GT 也失败** → 更像 execution / contact 语义问题
- **GT 成功、vision 失败** → 更像 candidate / 检测 / target 语义问题

---

## 6. 这轮最推荐的几个命令

如果你现在就要快速复现本轮调试入口，优先跑这几条：

```powershell
conda activate robotic-robocasa-rl

python scripts/demo_GT_grasp.py --scenario cube
python scripts/checkVision.py --scenario cube
python scripts/checkVision.py --scenario all --summary-table
python scripts/demo_vision_grasp.py --scenario can --render
python scripts/demo_vision_grasp.py --scenario cube --render
```

---

## 7. 备注

- `docs/getting-started.md` 仍然是项目级环境与结构说明主文档
- `docs/task.md` 记录本轮具体进展、问题和下一步方向
- 本文件 `LHYstart.md` 只负责让当前这批诊断脚本“尽快跑起来”
