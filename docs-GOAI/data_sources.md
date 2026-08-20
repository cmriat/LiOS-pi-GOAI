# 微调数据与依赖来源说明

> GOAI 2026 初赛提交材料 · 双臂赛道

## 数据来源

**本项目的监督微调数据全部来自 GOAI 2026 官方固定数据集，未加入外部演示数据。**

| 项 | 值 |
|--|--|
| 数据集 | GOAI 2026 官方固定数据集 |
| 规模 | 1200 episodes / 12 任务（每任务严格 100 条）/ 592,432 帧 / 6.58 小时 @ 25fps |
| 格式 | LeRobot v3.0 joint（双臂关节空间演示） |
| 任务覆盖 | 12 个 Generalization 基础任务（训练布局均为 standard 型） |

微调阶段未加入自采、合成或其他外部演示数据。模型并非从零训练：初始化自
[Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
公开发布的 `gs://openpi-assets/checkpoints/pi05_base` checkpoint，并转换为本项目
使用的 PyTorch 权重格式。比赛微调数据与基础模型来源在此分别列明。

## 基础模型与开源框架依赖

| 依赖 | 用途 | 说明 |
|--|--|--|
| Pi0.5 base checkpoint | 模型初始化 | Physical Intelligence/openpi 公开权重（对象标识见上文）；不属于 GOAI 微调数据 |
| Pi05 模型架构 | 策略网络基础 | 离散状态输入 + flow matching 动作生成 |
| OpenPI / Pi 训练框架 | 训练管线 | DCP checkpoint、FSDP 多卡训练 |
| RoboDojo Benchmark | 评测环境 | Isaac Sim 5.1 仿真、任务定义与评分 |
| XPolicyLab | 策略接口协议 | v1.0.0 policy server 协议（WS） |
| Pixi | 环境与依赖锁定 | 可复现环境 |

## 权重与资源

- 定版 checkpoint：ComboA 12 任务训练 · step30849 · EMA 权重（基于官方数据训练产出）
- 本地验证资源：单个 policy server 使用 1 × RTX 4090（24G）
- 未使用第三方私有权重或外部付费数据
