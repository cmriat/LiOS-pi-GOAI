# GOAI 2026 双臂赛道提交

## GOAI real-robot finals

The six-task PiperX policy is available at [policy/Lion_Pi05](policy/Lion_Pi05/README.md). It uses the unchanged official XPolicyLab server and a separately delivered checkpoint.

```bash
pixi install --locked
# Set checkpoint_path in configs/goai_real/server.yaml.
# Keep the official XPolicyLab checkout beside this repository, or set GOAI_ROBODOJO.
pixi run serve_goai
```

This is the real-robot entry. The simulation submission and its historical
scores are documented below; they do not measure this policy's physical success.

---

> RoboDojo Generalization 维度 · 官方提交仓库
> 本地 native 同口径评测：**Score 25.11 / SR 18.1%**；相对 2026-08-13 官方榜单快照第 1 名（23.55）高 **1.56 分**

本仓库是 GOAI 2026 双臂赛道的官方提交代码：自包含推理链、policy 启动接口与评测文档。评测方在官方环境本地运行本仓库 + 随附 checkpoint 即可评测。

## 提交物构成

| 项 | 位置 |
|--|--|
| 推理代码 | 本仓库（自包含，无外部私有依赖） |
| checkpoint / norm stats / manifest | `lion-vla-ckpt/`（经提交邮件单独提供下载方式，不入库；解压后保持相对结构） |
| tokenizer | `assets/paligemma_tokenizer.model`（内置，离线可用） |
| 评测说明 | `docs-GOAI/eval_launch.md` |
| 提交文档 | `docs-GOAI/`（数据说明 / 项目简介 / 技术方案） |

## 目录结构

```
.
├── src/pi/
│   ├── inference/           # GOAI 推理链：协议编解码、观测预处理、policy、归一化
│   ├── models/              # PaliGemma / Gemma / SigLIP transformers fork
│   └── models_pytorch/      # Pi05 PyTorch 实现（训练定版结构）
├── scripts/inference/goai/
│   ├── sim_server.py        # WS policy server（v1.0.0 CALL + 兼容旧版 infer）
│   ├── mock_goai_client.py  # 协议冒烟客户端（自测用）
│   └── sanitize_manifest.py # checkpoint manifest 脱敏工具
├── policy/pi05_goai/        # RoboDojo policy 目录接口（启动脚本 / eval 入口 / 元数据）
├── assets/                  # 内置 tokenizer
├── docs-GOAI/               # 比赛提交文档（数据 / 简介 / 技术方案 / 评测启动方式）
├── pixi.toml / pixi.lock    # 环境定义（goai-inference：Python 3.10 / PyTorch 2.7.1 CUDA 12.9）
```

## 快速开始

```bash
# 安装环境（一次性；锁定 Python 3.10 / PyTorch 2.7.1 CUDA 12.9 / Transformers 4.53.2）
pixi install -e goai-inference

# 启动 policy server（默认策略配置：action-horizon 8 / num-steps 20）
pixi run -e goai-inference python scripts/inference/goai/sim_server.py \
  --checkpoint lion-vla-ckpt/ema \
  --norm-stats lion-vla-ckpt/norm_stats_pt.json \
  --task-name stack_bowls \
  --config-name pi05_goai_joint \
  --host 0.0.0.0 --port 28000 \
  --device cuda:0 \
  --norm-mode per-timestamp \
  --apply-delta --seed 0

# 健康检查
curl http://127.0.0.1:28000/healthz
```

详细安装、启动与自测说明见 `docs-GOAI/eval_launch.md`。

## 协议

实现 **XPolicyLab v1.0.0**（兼容旧版 infer）：

```
hello → prepare_case → reset → 逐步 call(update_obs) → call(get_action) → trial_end
```

- `get_action` 返回 8 步 × 14 维**绝对关节位置** chunk（手臂 = delta + 当前状态，夹爪 clip [0,1]）
- 每条连接独立 session（RNG / 任务绑定隔离）；多环境并发由评测端以**每环境一条连接**管理，服务端未实现多环境合并的 batch 前向（详见 `docs-GOAI/eval_launch.md` §7）
- `action_case_id` 格式：`<task>_case` / `<task>_random_case`（12 任务表见 `docs-GOAI/eval_launch.md`）

## 验证成绩

| 口径 | 结果 |
|--|--|
| 本地 native 定版（24 配置 × 3 seeds × 25 episodes） | **Score 25.11 / SR 18.1%**，0 failures |

详见 `docs-GOAI/technical_solution.md`。

## 评测端参考

以下命令在 **RoboDojo 仓库根目录**执行，不属于本提交仓库的安装步骤。评测端
需先将本仓库的 `policy/pi05_goai` adapter 安装为
`XPolicyLab/policy/pi05_goai`，再连接已经启动的 policy server：

```bash
bash scripts/robodojo.sh client \
  --task stack_bowls \
  --policy-name pi05_goai \
  --policy-host <POLICY_SERVER_HOST> \
  --policy-port 28000 \
  --env-cfg arx_x5 \
  --seed 0 \
  --env-gpu 0 \
  --ckpt goai-submission \
  --action-type joint \
  --eval-num 1
```

`--ckpt` 在远程 client 模式中仅用于结果目录标识；实际 checkpoint 由 policy
server 启动参数决定。官方评测器若已有等价的 XPolicyLab client，可直接按上文
v1.0.0 协议连接，无需使用该包装命令。

## License

Apache 2.0（见 LICENSE）。
