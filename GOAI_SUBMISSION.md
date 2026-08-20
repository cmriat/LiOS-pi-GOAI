# GOAI 2026 Submission — 评测推理说明

本文件面向评测方：如何在本仓库 + 随附 checkpoint 上启动并评测我们的策略。

## 1. 提交物构成

| 项 | 说明 |
|--|--|
| 代码 | 本仓库（推理链 + policy 启动接口，自包含） |
| checkpoint | `lion-vla-ckpt/ema`（torch DCP 格式）|
| norm stats | `lion-vla-ckpt/norm_stats_pt.json` |
| 训练契约 manifest | `lion-vla-ckpt/norm_stats_manifest.json`（启动强校验必读，已脱敏）|
| tokenizer | 已内置于 `assets/paligemma_tokenizer.model`，无需联网 |

三者均位于 `lion-vla-ckpt/`（`ema` 的父目录），打包时请保持相对结构。

## 2. 环境安装（一次性）

```bash
pixi install -e goai-inference
```

环境锁定：Python 3.10 / PyTorch 2.7.1（CUDA 12.9）/ Transformers 4.53.2。

## 3. 启动 policy server

```bash
pixi run -e goai-inference python scripts/inference/goai/sim_server.py \
  --checkpoint lion-vla-ckpt/ema \
  --norm-stats lion-vla-ckpt/norm_stats_pt.json \
  --task-name stack_bowls \
  --config-name pi05_goai_joint \
  --host 0.0.0.0 --port 28000 \
  --device cuda:0 \
  --norm-mode per-timestamp \
  --apply-delta --seed 0
```

亦可经 RoboDojo policy 目录接口启动（见 `policy/pi05_goai/README.md`）。

**默认即定版配置**：`--action-horizon 8`、`--num-steps 20` 无需显式传参；
`--apply-delta` 为显式必传。服务端加载 checkpoint 时强校验训练契约
（动作归一化模式 / horizon / 任务表），配置不一致会拒绝启动——这是特性。

## 4. 健康检查

```bash
curl http://127.0.0.1:28000/healthz
```

返回 JSON（状态 / 运行时长 / 模型配置 / 最近绑定任务 / 推理统计）。

## 5. 协议

实现 XPolicyLab v1.0.0 协议（兼容旧版 infer）：

```
hello → prepare_case → reset → 逐步 call(update_obs) → call(get_action) → trial_end
```

- `get_action` 返回 8 步 × 14 维**绝对关节位置** chunk（手臂 = delta + 当前状态，夹爪 clip [0,1]）
- `action_case_id` 格式：`<task>_case` / `<task>_random_case`
- 12 任务表：arrange_largest_number, fold_clothes, hang_mugs, make_toast,
  pack_objects_into_box, pour_liquid_into_cup, push_T,
  sort_nesting_dolls_by_size, stack_blocks, stack_bowls,
  store_laptop_and_headphones, sweep_blocks
> 每次 `get_action` 前必须先发送当前帧 `update_obs`；服务端使用该连接
> 最近一次 observation 推理。请勿省略两者之间的协议调用，也请勿合并/抽帧发送。

## 6. 自测建议

仓库自带协议冒烟客户端（无需仿真环境）：

```bash
PYTHONPATH=src pixi run -e goai-inference python scripts/inference/goai/mock_goai_client.py \
  --url ws://127.0.0.1:28000 --expected-horizon 8
```

输出 `GOAI_MOCK_OK` 即协议通路正常。
