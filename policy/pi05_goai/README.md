# pi05_goai policy 目录（RoboDojo 约定）

GOAI 2026 双臂赛道提交的 policy 目录接口，自包含：启动本仓库自带的 GOAI
policy server，不依赖外部 Pi 仓库。

## 文件

| 文件 | 作用 |
|--|--|
| `setup_eval_policy_server.sh` | 启动 WS policy server（v1.0.0 CALL + 兼容旧版 infer） |
| `eval.sh` | RoboDojo policy 目录评测入口（转发到 setup 脚本） |
| `deploy.py` | Episode 执行循环（update_obs → get_action → take_action） |
| `deploy.yml` | Policy 元数据（name / action_type / embodiment） |

## 启动

```bash
bash policy/pi05_goai/setup_eval_policy_server.sh \
  RoboDojo stack_bowls \
  lion-vla-ckpt/ema \
  arx_x5 joint 0 0 pi05-goai 28000 0.0.0.0
```

参数依次为：bench / 任务名 / checkpoint 路径 / 环境配置 / 动作类型 /
seed / policy GPU / policy 环境名 / 端口 / 监听地址。

脚本默认即定版配置：action-horizon 8 / num-steps 20 / apply-delta
（per-timestamp 归一化）。协议与 checkpoint 布局见仓库根目录
`scripts/inference/goai/README.md` 与 `docs-GOAI/eval_launch.md`。
