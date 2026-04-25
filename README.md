这是针对 **V3.0 Final** 版本进行的最后两项关键优化补充。至此，你的 `guide.md` 已经完全覆盖了从物理公式、逻辑架构到跨进程通信和实验分析的所有细节。

以下是最终补齐后的完整版，可以直接作为你们项目的**官方开发手册**。

---

# 🤖 语义驱动双臂协作桌面整理系统 - 开发执行手册 (V3.1 Final)

## 1. 系统架构与 3D 感知协议

### 1.1 3D 坐标反投影参考实现 (数学对齐)
感知组必须确保 `pos` 的计算符合以下针孔相机模型变换。

**核心公式：**
给定像素坐标 $(u, v)$ 和深度 $d$，计算相机坐标系下的 $P_{cam} = [X_c, Y_c, Z_c]^T$：
$$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$
随后利用外参矩阵 $T_{world\_cam}$ 变换至世界坐标系：
$$P_{world} = R_{wc} \cdot P_{cam} + t_{wc}$$

### 1.2 异步感知与 Queue 通信 (规范对齐)
为保证 FSM 逻辑不被重型感知模型（GroundingDINO）阻塞，采用 `multiprocessing.Queue` 的异步通信方案（队列名固定为 `perception_queue`）。

**参考实现架构：**
```python
from multiprocessing import Queue
from queue import Empty

# 全局队列命名约定
perception_queue = Queue(maxsize=8)

# --- 感知组 (写入进程) ---
detected_objects = {
    "target": {"label": "knife", "pos": [x, y, z], "conf": 0.95, "timestamp": ts},
    "obstacles": [{"label": "cup", "pos": [x, y, z], "id": 101}],
    "status": "ready",
}
perception_queue.put(detected_objects)

# --- 逻辑组 (读取进程) ---
try:
    detected_objects = perception_queue.get(timeout=0.2)
except Empty:
    # stale data -> FSM 进入 RETRY_SENSING
    detected_objects = None
```

---

## 2. 增强型有限状态机 (FSM) 与降级逻辑

### 2.1 状态转换表 (含降级路径)

| 状态 | 转移条件 | 异常处理 (Exception) | 降级路径 (Fallback) |
| :--- | :--- | :--- | :--- |
| **IDLE** | 接收指令 -> **PLANNING** | - | - |
| **PLANNING** | 目标锁定 -> **CLEARING** | 目标丢失 -> **RETRY_SENSING** | 3次丢失 -> **FAILED** -> **IDLE** |
| **CLEARING** | 助手臂完成 -> **PLANNING** | 障碍物未动 -> **RETRY_PUSH** | 2次无效 -> **FAILED** -> **RESET** |
| **GRASPING** | 夹爪闭合 -> **VERIFYING** | 机械臂碰撞 -> **EMERGENCY** | - |
| **VERIFYING** | 抓取成功 -> **SUCCESS** | 抓空 (Width < limit) -> **RETRY_GRASP**| 2次抓空 -> **FAILED** -> **IDLE** |

---

## 3. 统一接口标准 (Standard Protocols)

### 3.1 3D 感知字典定义
```python
detected_objects = {
    "target": {"label": "knife", "pos": [x, y, z], "conf": 0.95, "timestamp": 1713200000.12},
    "obstacles": [{"label": "cup", "pos": [x, y, z], "id": 101}],
    "status": "ready" # 'ready', 'processing', 'error'
}
```

---

## 4. 实验评估与 Failure Mode 分类

为了在期末报告中提供更有深度的分析，实验组需按照下表统计失败原因：

### 4.1 故障模式分析表 (Failure Mode Analysis)
| 故障类别 | 判定条件 | 计数 (Count) | 优化方向 |
| :--- | :--- | :--- | :--- |
| **感知误差 (Perception Error)** | `RETRY_SENSING` 触发 | $N_1$ | 检查手眼标定、光照、遮挡逻辑 |
| **执行偏离 (Execution Drift)** | `RETRY_PUSH` 触发 | $N_2$ | 优化 OSC 控制参数或减小步长 |
| **物理滑落 (Physical Slip)** | `RETRY_GRASP` 触发 | $N_3$ | 增加摩擦力系数或调整抓取位姿 |

---

## 5. 开发里程碑 (Milestones)

1.  **T+3天 (Interface Demo)**：完成反投影数学模型校准，环境能同时渲染 RGB 和 Depth。
2.  **T+7天 (Vision-Loop)**：实现基于 `perception_queue` 的异步感知流，完成单臂抓取实验。
3.  **T+12天 (Final Evaluation)**：完成 RoboCasa 双臂协作任务测试，导出故障分类统计表。

---

## 6. 技术规约汇总
*   **坐标系**：统一 World Frame。
*   **并发控制**：感知与控制异步。
*   **延迟要求**：感知端到端 < 200ms。
*   **显存管理**：针对 8GB 显存，仿真分辨率限制在 512x512，开启推理模式。

---

### 🚀 今日行动建议
1.  **感知组**：将 `perception_queue` 写入逻辑集成到 GroundingDINO 推理脚本中。
2.  **执行组**：测试 `arm_safe_retract()`，确保任何时候调用都能让手臂瞬间回到初始待机点。
3.  **逻辑组**：在 FSM 中加入 `Counter` 变量，用于记录 Failure Mode 中的三类 RETRY 次数。

---
**手册版本：V3.1 (Final Release)**
**状态：已冻结，禁止未经全组讨论的重大接口变更。**
