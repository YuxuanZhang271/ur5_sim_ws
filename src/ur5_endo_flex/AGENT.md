# ur5_endo_flex

This is a integrated control system for `../ur5_endoscope` (called ***arm***) and `../endo_flex_control` (called ***flex***). 



- [ur5\_endo\_flex](#ur5_endo_flex)
  - [Model Integration](#model-integration)
  - [Control Scripts](#control-scripts)
  - [Control Panels](#control-panels)
  - [Constraints](#constraints)
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
    - Column 1 + Column 2: buttons for `HOME`, `RESET`, and `PULL OUT`
    - Column 3: tips on keyboard shortcut for `HOME`, `RESET`, and `PULL OUT`



## Constraints

1. Automatically adjust `wrist_1` of ***arm***. 
2. When RCM mode, the angle between `endoscope` and `fixed centrol axis` will not exceed 20deg. 
3. `MAX_BEND` of ***flex*** is 90deg. 



## Changes in Current Control Scripts

The main changes are to change `concrete control` to `continuous control`. That current movement control commands, you set a **fixed pose delta**, and triggered when you **press it once**. I'd like the continuous that control the movements by **press and hold** the keys or buttons, and there's a `velocity limit` for each joints. 