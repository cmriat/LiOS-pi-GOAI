# 项目简介

> GOAI 2026 初赛提交材料 · 双臂赛道

## 一句话

面向 RoboDojo 双臂通用操作任务的 Pi05 策略：官方数据抗过拟合重训 + 短动作块推理配置，本地 native 同口径 Score **25.11**；相对 **2026-08-13 官方榜单快照**第 1 名高 1.56 分。

## 项目内容

本项目以 Pi05 模型为基础，针对 GOAI 2026 双臂赛道（RoboDojo Generalization 维度，12 任务 × standard/random 布局共 24 配置）构建完整的「训练—评测—部署」方案：

- **训练**：以公开 Pi0.5 base checkpoint 初始化，监督微调仅使用官方 1200 条演示数据；通过 weight decay / EMA / LR 完整衰减的“正则包”抑制过拟合，12 任务均衡采样；五点 epoch 曲线定位最优步数（10 epoch）
- **推理**：8 步动作 chunk（action-horizon 8 / 20 步去噪 / delta 动作）；前代 checkpoint 的同权重消融中，相对 32/10 配置提升 +7.16 分
- **评测**：自建与官方同口径的 screening/native 两级评测体系，17 轮评测科学收敛到定版配置
- **部署**：自包含本地 policy server，支持官方 v1.0.0 CALL 协议、健康检查与按连接隔离的 session

## 成绩

| 指标 | 值 |
|--|--|
| 定版 native Score / SR | **25.11 / 18.1%**（1800 episodes，0 failures） |
| 2026-08-13 官方榜单快照第 1 名 | 23.55（[Xiaomi-Robotics-1，论文表 5](https://robotics.xiaomi.com/robot-static-resource/xiaomi-robotics-1/xiaomi-robotics-1.pdf)） |
| 相对该快照第 1 名 | **+1.56** |

## 亮点

1. **科学定版**：17 轮对照评测 + epoch 曲线五点定位 + ah/ns 扫描，每个配置决策都有数据支撑
2. **抗过拟合重训**：正则包使整条训练曲线上移 +2.40，而非头痛医头的早停
3. **微调数据透明**：监督微调仅使用官方数据，并明确披露公开 Pi0.5 基础权重来源
4. **工程可复现**：独立 Pixi 环境、训练契约强校验、标准 CALL 协议、健康检查和端到端复验闭环
