# Real-policy interface

## Ownership and checkpoint contract

The unchanged official XPolicyLab server discovers
`XPolicyLab.policy.Lion_Pi05.model.Model`. This repository implements the
model adapter and runtime. `deploy.py` re-exports official Pi_05 episode
callbacks. Hardware control, physical reset and camera acquisition belong to
the client and evaluator.

The checkpoint architecture uses 12 task slots, VLM task embeddings, discrete
VLM state, two embodiments with one token each, trained horizon 32,
per-timestamp quantile normalization, and arm delta targets. Startup checks
the manifest, statistics hash, tensor shapes and strict DCP coverage. Only
genuinely tied aliases may be omitted from EMA. All tasks share one model.

## Tasks

| Slot | Slug | Canonical instruction |
| --- | --- | --- |
| 0 | fill_pen_holder | Fill the pen holder |
| 1 | put_objects_into_basket | Put the objects into the basket |
| 2 | stack_and_cover_blocks | Stack and cover the blocks |
| 3 | stack_bowls | Stack the bowls |
| 4 | stand_up_bottles | Stand up the bottles |
| 5 | insert_charger | Insert the charger |

These six tasks use PiperX embodiment 0. The preliminary simulator task table
has different indices and must not be substituted.

Each observation may supply `instruction`, `prompt` or `task_name`; multiple
fields must resolve to the same task. Matching normalizes Unicode, case,
punctuation and whitespace. Exact matches precede nearest matches requiring
similarity >= 0.90. Nearest matches are logged; unsupported language fails.
Missing fields use the optional configured `task_name`, otherwise they fail.
Configured tasks must match exactly after normalization. Active sessions
cannot switch tasks until reset.

## Observations

`update_obs(obs)` accepts one mapping; `update_obs_batch(obs_list)` accepts a
nonempty list. `env_idx` is a nonnegative integer, defaulting to batch position.
Duplicate indices fail. Failed updates preserve the previous validated batch.

`state` is a finite 14-element vector or a mapping concatenated in this order:

| Key | Shape | Meaning |
| --- | --- | --- |
| left_arm_joint_state | (6,) | Left joint angles in training order |
| left_ee_joint_state | (1,) | Left normalized gripper opening |
| right_arm_joint_state | (6,) | Right joint angles in training order |
| right_ee_joint_state | (1,) | Right normalized gripper opening |

Decoded RGB images are uint8 `(480,640,3)` or `(3,480,640)` under `vision`
(or `images`). Camera values are arrays or mappings with `color` (or `rgb`).

| Model camera | Accepted keys |
| --- | --- |
| cam_high | cam_high, cam_head, head_camera, top_camera |
| cam_left_wrist | cam_left_wrist, left_wrist, left_camera, wrist_left |
| cam_right_wrist | cam_right_wrist, right_wrist, right_camera, wrist_right |

JPEG decoding belongs to official `decode_obs_images`; the adapter receives
decoded arrays. Extra state, camera and top-level metadata are ignored.
`images_preprocessed: true` fails. The pipeline applies trained bottom/right
square padding and model resizing once.

## Actions and lifecycle

`get_action()` requires one latest observation. `get_action_batch()` follows
the latest batch order or requested unique indices. Each result is a list of
`execution_horizon` dictionaries with the four state keys and shapes above.
Targets are absolute after inverse normalization and arm-delta addition.
Gripper openings are clipped to [0,1]. Client hardware conversion must match
training units and joint order. The nominal data interval is 0.04 s; eight
actions represent 0.32 s playback, excluding inference delays.

Model `reset()` clears all environments' observations, sessions, task bindings
and RNG state. `on_trial_end()` also resets. Each environment uses its own
seed derived from the configured seed and index. `prepare_case()` validates
the declared task; observations determine session tasks.

These are model methods. The official synchronous client uses `call(...)`:
`call("reset")`, `call("prepare_case", metadata)` and
`call("trial_end", result)` map to dedicated protocol messages. There are no
standalone `client.reset()` or `client.prepare_case()` methods.

## Optional postprocessing and validation limits

`postprocess.enabled` defaults to false. Raw gripper statistics are logged
even while disabled. When enabled, opening below `squeeze_below` is reduced
by `squeeze` and floored at zero. `per_task` can select a subtraction by task
name or slug. Arms are unchanged. Values are dimensionless opening ratios;
this is not force control or a guarantee of improved grasping.

Synthetic parity, reset, raw/JPEG and batch checks establish software behavior.
They do not establish physical success, camera calibration, grasp force or
complete compatibility with the organizer's final deployment environment.

## Deployment-client compatibility

The corresponding real client sends MessagePack binary frames with msgpack-numpy
compatible arrays. Its `PolicyWsClient` has convenience `reset`, `prepare_case`,
`trial_end`, `heartbeat` and `infer` methods; these are distinct from the official
`WsModelClient` API described above. `infer` carries `payload.observation` and
returns `payload.actions`; the two-call path uses `payload.func_name` and
`payload.obs`, returning `payload.result`. A reconnect repeats hello and verifies
`server_instance_id`. Hardware deployment supplies the canonical task instruction
and defaults to environment 0. Both paths use the same unchanged official server.


The six official full instructions are also exact aliases of slots 0–5:

1. Pick up the pen holder and place all the pens into it.
2. Place all the objects on the table into the basket.
3. Stack the blocks on the table, then cover them with the cup.
4. Stack the bowls on the table.
5. Stand the bottle upright.
6. Insert the charger plug into the power strip, then connect the charging cable to the plug.

These aliases preserve the existing checkpoints' embedding indices.

### Client deployment boundary

Passing the external transport validator does not validate its camera acquisition
or actuator conversion. The server requires normalized gripper ratios in both
directions and 0.04 s nominal playback. JPEG bytes must decode through the official
image decoder into RGB-order arrays; its OpenCV decoder does not swap channels.
The official debug encoder therefore encodes its RGB-order array directly.
A client that converts RGB to BGR before JPEG encoding needs an explicit correction
at the deployment boundary. A client that sends or executes gripper values in
meters must convert using its calibrated hardware mapping. No automatic unit or
color inference is performed by this policy.
