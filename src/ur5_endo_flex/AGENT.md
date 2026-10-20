# ur5_endo_flex

This is a integrated control system for `../ur5_endoscope` (called ***arm***) and `../endo_flex_control` (called ***flex***). 



- [ur5\_endo\_flex](#ur5_endo_flex)
  - [Model Integration](#model-integration)
  - [Control Scripts](#control-scripts)
  - [Control Panels](#control-panels)
  - [Constraints](#constraints)
  - [Target Tracking](#target-tracking)
  - [Changes in Current Control Scripts](#changes-in-current-control-scripts)




## Model Integration

The base of ***flex*** should be fixed to the endoscope of ***arm***, position from endoscope coordinate `(0, 0, -398)`mm, pose: 1. $\text{z-axis}_{flex} = -\text{z-axis}_{endoscope}$, 2. $\text{x-axis}_flex = \text{z-axis}_{tool0}$. 



## Control Scripts

1. Modes

- For ***arm***: 

| Mode | Description |
|---|---|
| Linear Mode | 1. Goal is to fix `tool0` orientation and control its position. <br> 2. The RCM point, central axis will move with the `endoscope`.  |
| RCM Mode | 1. Goal is to fix RCM point and centrol axis, then control `tool0` pose. <br> 2. The angle between endoscope and central axis will not exceed 20deg.  |

- For ***flex***: 

| Mode | Description |
|---|---|
| Free Mode | `base_yaw` rotate while `seg` not moving. |
| Fixed Mode | `base_yaw` and `seg` moves together, keep z-axis of `end_cylinder` pointing to a fixed direction. |

By default the modes should be `linear` and `fixed`. 

1. Keyboard Shortcuts

Please change keyboard shortcuts to the table below: 

- Basic movements: 

| Key | Control | Description |
|---|---|---|
| `UP` `DOWN` `LEFT` `RIGHT` `[` `]` | ***arm*** | 1. In linear mode: go up, go down, go left, go right, go forward, go backward <br> 2. In RCM mode: look up, look down, look left, look right, go in (forward), go out (backward) |
| `W` `S` `A` `D` | ***flex*** | 1. In free mode: straighten, bend, rotate left, rotate right <br> 2. In fixed mode: straighten, bend, twist left, twist right |

- Mode: 

| Key (press together) | Control | Description |
|---|---|---|
| `ctrl` + `M` | Arm | Toggle between linear mode and RCM mode (linear mode by default) |
| `ctrl` + `N` | Flex | Toggle between free mode and fixed mode (fixed mode by default) |

- Semi-automatic commands: 

| Key (press together) | Mode | Description |
|---|---|---|
| `ctrl` + `H` | Linear | 1. Set ***arm*** joint angles to `[0.0, -120.0, -120.0, -30.0, 90.0, 45.0]` <br> 2. For ***flex***, `base_yaw = 0.0`, `arc_len = ARC_MAX = 0.050`, `ext_yaw = 0.0`, `curvature = 0` |
| `ctrl` + `R` | Linear | 1. The functionality of `ctrl` + `H` <br> 2. Set modes to `linear` and `fixed` (default modes) |
| `ctrl` + `P` | RCM | 1. The `endoscope` rotate back to align the fixed centrol axis, the ***flex*** curvature to 0 <br> 2. Go out until the `end_cylinder` of ***flex*** is 10cm away behind the RCM point |



## Control Panels

The arrangement should be a 3x3 grid: 

1. Row 1
    - Column 1: real-time ***arm*** mode + ***arm*** joint angles state
    - Column 2: button toggle ***arm*** mode, buttons for `UP` `DOWN` `LEFT` `RIGHT` `[` `]`
    - Column 3: tips on keyboard shortcut for ***arm*** mode switch and movement control
2. Row 2
    - Column 1: real-time ***flex*** mode + ***flex*** 4 parameters + ***flex*** 4 tendon lengths
    - Column 2: button toggle ***flex*** mode, buttons for `W` `S` `A` `D`
    - Column 3: tips on keyboard shortcut for ***flex*** mode switch and movement control
3. Row 3
    - Column 1 + Column 2: target tracking buttons for `LN_No3`, `LN_No4d`, `LN_No6`, `LGA_No7` plus a `STOP` button, with live status (active target, camera-to-target distance, lock state)
4. Row 4
    - Column 1 + Column 2: buttons for `HOME`, `RESET`, and `PULL OUT`
    - Column 3: tips on keyboard shortcut for `HOME`, `RESET`, `PULL OUT`, and target tracking



## Constraints

1. Automatically adjust `wrist_1` of ***arm***. 
2. When RCM mode, the angle between `endoscope` and `fixed centrol axis` will not exceed 20deg. 
3. `MAX_BEND` of ***flex*** is 90deg. 



## Target Tracking

- Each per-target button engages a closed-loop **hybrid RCM servo** (20 Hz) that keeps the distal camera looking at that scene target while holding `tracking_standoff_m` (default 10.5 cm) from it; dynamic targets such as `LGA_No7` are followed continuously.
- **RCM-only gating:** tracking is refused (with a GUI status reason) unless the arm is in RCM mode with a captured RCM point (`Ctrl+M`) **and** the flex is in `fixed` mode (`Ctrl+N`). Disabling RCM or leaving `fixed` mid-track auto-cancels tracking with a log.
- **Aim convention:** the camera looks along `R_end[:, 2]` (flex-end `+z`), which equals the **negative** shaft axis while the endoscope is straight. The servo drives the virtual aim point `p_aim = p_cam + standoff · ẑ_cam` onto the target, which satisfies look-at and standoff simultaneously.
- **Zone schedule** (on the flex bend `θ_total = κ · L_eff`, `L_eff = 4.2 cm`):

| Zone | `θ_total` | Flex weight | Arm tilt weight | Behavior |
|---|---|---|---|---|
| 1 | ≤ 50° | 1 | 10⁶ (locked) | Flex only; axial depth still routes to insertion `d_ins` |
| 2 | 50°–80° | `1 + 20s²` | `1 + (10⁶−1)(1−s)²` | Smooth blending (`s = (θ−50°)/30°`) as the flex approaches saturation |
| 3 | ≥ 80° | 1, or 10⁴ if outward | 1 | Arm takeover; outward bending is locked to the arm, inward unbending stays flex-first |

- Any manual movement key cancels tracking immediately and hands control back to the normal press-and-hold bindings.
- `■ STOP` disengages tracking and holds the current ***arm*** / ***flex*** state (no retract or return motion).



## Changes in Current Control Scripts

The main changes are to change `concrete control` to `continuous control`. That current movement control commands, you set a **fixed pose delta**, and triggered when you **press it once**. I'd like the continuous that control the movements by **press and hold** the keys or buttons, and there's a `velocity limit` for each joints. 