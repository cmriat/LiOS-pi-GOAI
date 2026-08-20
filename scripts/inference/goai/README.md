# GOAI 2026 提交推理服务

GOAI 2026 双臂赛道（RoboDojo Generalization 维度）的 policy server：自包含
WebSocket 服务，实现 XPolicyLab v1.0.0 协议（CALL 帧 + 兼容旧版 infer），
并提供运维用 `/healthz` 端点。

## 文件

| 路径 | 作用 |
|--|--|
| `scripts/inference/goai/sim_server.py` | WS server：协议分发、按连接 session/RNG、healthz |
| `src/pi/inference/goai_sim_policy.py` | Policy：checkpoint 契约校验、观测预处理、推理 |
| `src/pi/inference/goai_observation.py` | GOAI 相机映射 + 图像预处理 |
| `src/pi/inference/goai_helpers.py` | 自包含 manifest/归一化工具（无 B1K 依赖） |

## Checkpoint

基于 GOAI 2026 官方数据集（1200 episodes / 12 任务）训练。
最终提交 checkpoint：`lion-vla-ckpt/ema`（torch DCP 格式）；其父目录
`lion-vla-ckpt/` 附带 `norm_stats_pt.json` 与 `norm_stats_manifest.json`。

## 环境

提供最小化推理环境（无训练依赖）：

```bash
pixi install -e goai-inference
pixi run -e goai-inference python scripts/inference/goai/sim_server.py ...
```

## 启动

```bash
pixi run -e goai-inference python scripts/inference/goai/sim_server.py \
  --checkpoint lion-vla-ckpt/ema \
  --norm-stats lion-vla-ckpt/norm_stats_pt.json \
  --task-name stack_bowls \
  --config-name pi05_goai_joint \
  --host 0.0.0.0 --port 28000 \
  --device cuda:0 \
  --action-horizon 8 --num-steps 20 \
  --compile-mode none \
  --norm-mode per-timestamp \
  --apply-delta --seed 0 --warmup
```

健康检查：

```bash
curl http://127.0.0.1:28000/healthz
```

## 离线说明

- paligemma tokenizer 内置于 `assets/paligemma_tokenizer.model`，默认从仓库
  加载（无需联网），可用 `GOAI_TOKENIZER_PATH` 覆盖。
- 打包 checkpoint 前先脱敏 manifest 中的训练机路径：
  `python scripts/inference/goai/sanitize_manifest.py lion-vla-ckpt/norm_stats_manifest.json`
- server 默认即定版配置：`--action-horizon 8`、`--num-steps 20` 无需显式传参；
  `--apply-delta` 为显式必传。无需环境变量注入。

## 协议

- v1.0.0 CALL：`hello` → `prepare_case` → `reset` → 逐步 `call(update_obs)` →
  `call(get_action)` → `trial_end`。`get_action` 返回 8 步 × 14 维绝对关节
  位置 chunk（手臂 = delta + 当前状态，夹爪 clip [0,1]）。
- 兼容旧版 `infer` 帧。
- `action_case_id` 必须为 `<task>_case` 或 `<task>_random_case`（12 任务表：
  arrange_largest_number, fold_clothes, hang_mugs, make_toast,
  pack_objects_into_box, pour_liquid_into_cup, push_T,
  sort_nesting_dolls_by_size, stack_blocks, stack_bowls,
  store_laptop_and_headphones, sweep_blocks）。
- 多环境并发与 batch 调用说明见 `docs-GOAI/eval_launch.md` §7。
