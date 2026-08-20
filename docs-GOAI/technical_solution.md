# 技术方案

> GOAI 2026 初赛提交材料 · 双臂赛道（RoboDojo Generalization 维度）
> 本地 native 同口径评测：**Score 25.11 / SR 18.1%**；相对 2026-08-13 官方榜单快照第 1 名（23.55）高 1.56 分

## 1. 方案总览

- **模型**：公开 Pi0.5 base checkpoint 初始化的 Pi05（离散状态输入 + flow matching 动作生成）双手机器人策略
- **定版 checkpoint**：12 任务 ComboA 训练 · step30849（10 epoch）· EMA 权重
- **推理配置**：action-horizon 8（闭环比）/ 去噪步数 20 / delta 动作 / per-timestamp 归一化
- **评测协议**：v1.0.0 CALL（update_obs + get_action 逐步闭环）

实验演进分为两步：前代 checkpoint 上，推理配置由 32/10 调整为 8/20，
Score 从 15.55 提升至 22.71；随后保持 8/20，采用 ComboA 训练 recipe，
最终 checkpoint 达到 25.11。两步来自不同的受控对照，不将 +7.16 归因于
最终 ComboA checkpoint 的未测配置。

## 2. 模型与推理管线

- **观测**：3 相机 RGB（480×640，pad 至 640×640 → 224×224）+ 14 维双手机械臂 proprio（左臂 6D + 左夹 1D + 右臂 6D + 右夹 1D）
- **状态表示**：离散状态输入（Paligemma 词表 token 化）+ 12 任务 task embedding（standard 与 `_random` 布局共用同一技能 id）
- **动作生成**：flow matching 去噪（20 步），输出 8 步 × 14 维关节 delta chunk；执行侧将 delta 加回当前关节状态、夹爪绝对量 clip [0,1]，开环执行 8 步后取下一 chunk
- **归一化**：quantile 归一化 + per-timestamp 动作归一化（与训练完全同口径，ckpt 契约强制校验）

## 3. 训练方案（抗过拟合 ComboA）

新 recipe 相对旧版的三项改动（正则包）：

| 项 | 值 | 作用 |
|--|--|--|
| weight decay | 0.05 | 抑制权重增长 |
| EMA | 0.9995 | 权重平滑 |
| LR decay floor | 0 | 学习率完整衰减到底 |

- 采样：task/episode 均衡采样（12 任务公平覆盖）
- 数据：监督微调仅使用**官方固定数据集**（1200 episodes / 12 任务 / 592,432 帧，LeRobot v3.0 joint 格式），未加入外部演示数据；模型初始化使用公开 Pi0.5 base checkpoint
- 效果：正则包使整条 epoch 曲线**上移 +2.40 分**（22.71 → 25.11），而非推迟过拟合点；五点曲线单峰验证 10 epoch（30849 步）即最优步数，「续训右移」假设证伪

## 4. 推理配置（闭环比）

官方复评路径不注入环境变量，服务端默认配置即定版：

- `--action-horizon 8` / `--num-steps 20` / `--apply-delta`；提交 server 的 `--compile-mode` 默认 `none`
- 8/20 相对 32/10：**前代 0804 checkpoint 同权重实测 +7.16 分**（22.71 vs 15.55）；最终 ComboA checkpoint 未做 32/10 对照
- ns40（更深闭环）两点 native 复测 23.40 / 23.11，收益饱和——维持 ah8/ns20
- `torch.compile` 仅作为可选吞吐优化，不属于分数配置，也不改变 checkpoint 或动作语义

## 5. 评测与验证

- 自建评测体系与官方同口径：24 配置（12 任务 × standard/_random）× 3 seeds，screening（5 ep）快速筛选 + native（25 ep）定版
- 17 轮评测收敛到定版：native 1800 episodes 完赛 **0 failures**
- 独立提交仓库使用 2 个 server + 14 个独立仿真实例完成 72 episodes 协议回归：72/72 jobs、0 failures
- 另一次 v1.0.0 CALL 同口径 72 episodes 复验同样为 0 failures；小样本结果仅用于协议回归，不替代 native 成绩

## 6. 关键数字

| 指标 | 值 |
|--|--|
| 定版 native Score / SR | **25.11 / 18.1%** |
| 对 2026-08-13 官方榜单快照 #1（[Xiaomi-Robotics-1 23.55，论文表 5](https://robotics.xiaomi.com/robot-static-resource/xiaomi-robotics-1/xiaomi-robotics-1.pdf)） | **+1.56** |
| 前代 checkpoint 32/10 → 8/20 增益 | +7.16 |
| 抗过拟合重训增益 | +2.40 |
