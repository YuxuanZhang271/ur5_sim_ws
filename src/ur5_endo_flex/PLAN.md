# `ur5_endo_flex` — Background, Diagnosis, and Plan

> Status: **investigation done, no source changes applied yet.**
> Captures: the project context, the two visible problems reported by the user, the test
> run, the root-cause analysis, and the agreed minimum-diff fix plan.

---

## 1. Project context

`ur5_endo_flex` is an integrated control stack that combines two vendored subsystems:

| Subsystem | Package (vendored under `src/`) | Role |
|---|---|---|
| **arm** | `ur5_endoscope` (UR5 + rigid endoscope tube) | 6-DOF arm with a long endoscope that has a fixed camera at its tip |
| **flex** | `endo_flex_control` (continuum segment chain) | 8-segment tendon-driven continuum attached to the endoscope tip |
| **integrator** | `ur5_endo_flex` (this package) | One launch file, one URDF, one Tkinter control panel driving both |

Launch entry point: `ros2 launch ur5_endo_flex ur5_sim.launch.py`
Driver: `ur_simulation_gz` (UR's official gz-sim launch), passed a custom xacro
description (`ur5_endo_flex.urdf.xacro`).

URDF composition (top-down):

```
world
└── supporter_link               (0.10 m × 0.80 m grey cylinder, static)
    └── base_link … tool0        (UR5 macro from ur_description)
        └── endoscope_link       (package://ur5_endoscope/meshes/endoscope_mid.stl)
            └── flex_base_link   (fixed mount, see endo_flex_flexbase.xacro)
                └── start_cylinder → yaw_base (base_yaw revolute)
                    └── segment_1
                        └── 7× {ext_yaw_N, ext_pitch_N, ext_yaw_back_N, extension_N} → segment_{N+1}
                            └── end_cylinder
                                └── camera_link → camera_optical_frame
```

Controller chain (only the 6 UR joints are wired to `gz_ros2_control`):

```xml
<ros2_control name="ur5_with_endoscope" type="system">
  <plugin>gz_ros2_control/GazeboSimSystem</plugin>
  <!-- shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3: position cmd + pos/vel state -->
</ros2_control>
```

The 28 flex joints are **not** in this block.

---

## 2. Reported problems

1. **No robot renders in Gazebo.** RViz shows the full model fine. The Gazebo window
   stays effectively empty.
2. **The camera in RViz swings/rotates** even when the UR5 is supposed to be holding
   still. The camera is fixed at the end of the flex; the user expected it to be
   stable.

---

## 3. Test run

Commands executed (as the user specified):

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install     # 6 packages, all clean
source install/setup.bash
ros2 launch ur5_endo_flex ur5_sim.launch.py
```

Full log preserved at `/tmp/ur5_logs/launch.log` (524 lines).

### Key observations from the log

| Symptom | Evidence |
|---|---|
| Mesh URIs unresolved | `Unable to find file with URI [model://ur_description/meshes/ur5/visual/base.dae]` … repeated for `base/shoulder/upperarm/forearm/wrist{1,2,3}.dae` and `collision/*.stl`, then `model://ur5_endoscope/meshes/endoscope_mid.stl`. Each followed by `Failed to load geometry for visual: <link>_visual`. |
| Flex segment "spread" attempt silently ignored | `Warning [Utils.cc:132] [joint[@name="extension_N"]/initial_position …]: XML Element[initial_position], child of element[joint], not defined in SDF. Copying[initial_position] as children of [joint].` |
| Camera frame id also silently ignored | Same warning for `gz_frame_id` under the `<sensor>` (legacy Gazebo-Classic idiom). |
| `gz_ros2_control` only sees the 6 UR joints | `[INFO] [gz_ros_control]: System Successfully configured!` then 6 joint names listed. None of `ext_yaw_N / ext_pitch_N / ext_yaw_back_N / extension_N / base_yaw` are reported as state/command interfaces. |
| `ros_gz_sim create` did spawn the entity | `[ros_gz_sim]: Entity creation successful.` — Gazebo has the kinematic tree, just no visible geometry. |

### Other notes (out of scope per user)
- `control_panel_gui.py` and `initial_pose_publisher.py` died on launch with
  `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`. Root cause:
  `which python3` returns `/home/yuxuanzhang/Documents/Workspace/miniconda3/bin/python3`
  (Python 3.14) even after `conda deactivate`, and rclpy is built for 3.12.
  The user has confirmed this is handled on their side and asked us to ignore it.

---

## 4. Root cause

### Issue 1 — empty Gazebo scene
UR's `ur_macro.xacro` emits visual/collision mesh URIs in SDF model-DB form:

```xml
<mesh filename="model://ur_description/meshes/ur5/visual/base.dae"/>
```

`ros_gz_sim create` converts the URDF to SDF on the fly; the conversion
**rewrites `package://` to `model://`** for compatibility. `gz-sim`'s
`SystemPaths` resolves `model://X/Y` by walking every directory in
`GZ_SIM_RESOURCE_PATH` looking for one literally named `X` that contains a
`model.config`. Neither `ur_description` nor `ur5_endoscope` ships a
`model.config`, so every visual — and the endoscope's only visual — silently
fails to load. The supporter (a primitive `<cylinder>`) is the only thing
that survives, which matches the user's "no robot" observation.

RViz renders the same URDF fine because RViz's `package://` resolver uses
ROS package paths, not gz's SDF model database.

### Issue 2 — camera swing
Inside `endo_flex_flexbase.xacro`, the `<xacro:macro name="seg_joint">` declares
**4 dynamic joints per segment** (28 total: `ext_yaw_N`, `ext_pitch_N`,
`ext_yaw_back_N` revolute, and `extension_N` prismatic). None of them appear in
the `<ros2_control>` block. In `dartsim`:

- They have no command interface → no actuator holds them.
- URDF default is `0` for each → all 8 segments collapse onto `yaw_base` at
  start-up (the `<gazebo reference="extension_N"><initial_position>SEG_LEN` was
  a Gazebo Classic idiom that gz-sim just copies-and-ignores; the log warning
  confirms it).
- Gravity + parent-chain motion + numerical drift in the unactuated DOFs →
  the chain oscillates. RViz reflects whatever `joint_state_broadcaster`
  reports (just the 6 UR joints), so the camera position drifts in the
  rendered TF tree.

The user's hypothesis ("no physical sim for the flex segments") is correct.

---

## 5. Plan (minimum diff, no changes yet)

The two fixes touch **6 files** (4 edits, 2 new). Nothing in `ur5_endo_flex/scripts/`,
no controllers YAML change, no fork of `ur_description`'s xacro.

### 5.1 Issue 1 — register mesh packages as gz-sim models

| # | File | Op | Why |
|---|---|---|---|
| 1a | `src/Universal_Robots_ROS2_Description/model.config` | NEW | Tells gz-sim "model root is this package" |
| 1b | `src/Universal_Robots_ROS2_Description/CMakeLists.txt` | EDIT (+1 line) | `install(FILES model.config …)` so colcon ships it |
| 1c | `src/ur5_endoscope/model.config` | NEW | Same for endoscope mesh |
| 1d | `src/ur5_endoscope/CMakeLists.txt` | EDIT (+1 line) | Same `install(FILES …)` |
| 1e | `src/ur5_endo_flex/launch/ur5_sim.launch.py` | EDIT (+6 lines) | `SetEnvironmentVariable` to put both share dirs on `GZ_SIM_RESOURCE_PATH` |

**`model.config` template** (used twice with different `<name>`):
```xml
<?xml version="1.0"?>
<model>
  <name>ur_description</name>            <!-- or ur5_endoscope -->
  <version>1.0.0</version>
  <sdf version="1.7">ur_description.sdf</sdf>
  <description>URI-only shim: makes model://<name> resolve to the <name> package root.</description>
</model>
```

> gz-sim only needs `model.config` for `model://` URI lookup; a real `model.sdf`
> is unnecessary because nothing `<include>`s the package as a model.

**Launch-file change** (added once, near the top of `generate_launch_description`):
```python
from launch.actions import SetEnvironmentVariable
...
gz_resource_path = ':'.join(filter(None, [
    os.path.join(get_package_share_directory('ur_description')),
    os.path.join(get_package_share_directory('ur5_endoscope')),
    os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
]))
```
And in the returned `LaunchDescription`:
```python
SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=gz_resource_path),
```

### 5.2 Issue 2 — make the flex chain rigid (no swing)

One file, one macro body rewrite.

| # | File | Op | Why |
|---|---|---|---|
| 2 | `src/ur5_endo_flex/urdf/endo_flex_flexbase.xacro` | EDIT inside `<xacro:macro name="seg_joint">` | Convert 4 joint types to `fixed`, drop the misplaced `<gazebo><initial_position>` block, bake `SEG_LEN` offset into the `extension_${idx}` joint origin so segments stand along z |

Concretely, replace the four joint blocks (and the `<gazebo reference="extension_${idx}">` block) with:

```xml
<joint name="ext_yaw_${idx}"      type="fixed">
  <origin xyz="0 0 ${origin_z}"   rpy="0 0 0"/>
  <parent link="${parent}"/>
  <child  link="ext_yaw_mid_${idx}"/>
</joint>
<joint name="ext_pitch_${idx}"    type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="ext_yaw_mid_${idx}"/>
  <child  link="ext_pitch_mid_${idx}"/>
</joint>
<joint name="ext_yaw_back_${idx}" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="ext_pitch_mid_${idx}"/>
  <child  link="ext_yaw_back_mid_${idx}"/>
</joint>
<joint name="extension_${idx}"    type="fixed">
  <origin xyz="0 0 ${SEG_LEN}" rpy="0 0 0"/>
  <parent link="ext_yaw_back_mid_${idx}"/>
  <child  link="segment_${idx+1}"/>
</joint>
```

The stub `ext_yaw_mid_*`, `ext_pitch_mid_*`, `ext_yaw_back_mid_*` link definitions
stay (kept for compatibility with existing kinematics code). Their `<inertial>` is
harmless at this scale.

**Why this fixes the swing:**
- `FIXED` joints have no axis, no limit, no DOF → `dartsim` treats them as rigid
  kinematic constraints.
- No source of motion → no force to flap against. Gravity still acts on the
  pole but rigid transmission just adds a bending moment at the UR5 mount,
  not a camera wobble.
- Original joint names preserved → your kinematics scripts that look up
  `ext_yaw_N` / `ext_pitch_N` / `ext_yaw_back_N` / `extension_N` still
  resolve (they'll just read `0`).
- The flex visual becomes a straight 7-segment pole of 8 mm-spaced cylinders
  + red `end_cylinder` + camera — matches the RViz look minus the swing.

**Optional belt-and-suspenders** (same xacro): set `mass="0"` on
`start_cylinder`, every `segment_*`, `end_cylinder`, `camera_link`,
`camera_optical_frame`, and all `tendon_*_link` blocks. With `FIXED` joints
this is purely cosmetic insurance against any future solver jitter.

### 5.3 Files explicitly NOT touched

- `src/ur5_endo_flex/scripts/*.py` (rclpy issue, user-side)
- `src/ur5_endo_flex/urdf/ur5_endo_flex.urdf.xacro` (no edits needed)
- `src/ur5_endoscope/urdf/*.xacro`
- `src/Universal_Robots_ROS2_Description/urdf/`
- `src/endo_flex_control/**`
- `src/Universal_Robots_ROS2_GZ_Simulation/**` (vendored `ur_simulation_gz`)
- `ur_controllers.yaml` / any controller config

---

## 6. Diff summary

| File | Operation | LoC | Reason |
|---|---|---|---|
| `src/Universal_Robots_ROS2_Description/model.config` | NEW | +6 | Registers UR mesh DB with gz-sim |
| `src/Universal_Robots_ROS2_Description/CMakeLists.txt` | EDIT | +1 | Ship the marker file |
| `src/ur5_endoscope/model.config` | NEW | +6 | Same for endoscope mesh |
| `src/ur5_endoscope/CMakeLists.txt` | EDIT | +1 | Same |
| `src/ur5_endo_flex/launch/ur5_sim.launch.py` | EDIT | +6 | Set `GZ_SIM_RESOURCE_PATH` |
| `src/ur5_endo_flex/urdf/endo_flex_flexbase.xacro` | EDIT | refactor 1 macro | Make flex chain rigid |

---

## 7. Verification (to run after the diff is applied)

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch ur5_endo_flex ur5_sim.launch.py
```

Then re-grep `/tmp/ur5_logs/launch.log`:

- **Should disappear**:
  - `Unable to find file with URI [model://ur_description/...]`
  - `Unable to find file with URI [model://ur5_endoscope/...]`
  - `Failed to load geometry for visual: …`
  - `XML Element[initial_position], child of element[joint], not defined in SDF`
- **Should remain** (expected / harmless): `Qt binding-loop`, `libEGL warning: egl: failed to create dri2 screen`
  (NVIDIA EGL fallback, normal in headless/containerized environments),
  `Mesh construction from an SDF has not been implemented yet for dartsim` (collisions
  use cylinder primitives, not meshes — fine).

Visual acceptance:
- Gazebo: full UR5 DAE meshes + grey endoscope tube + 8 blue flex cylinders + red
  `end_cylinder` + camera lens, all on top of the grey supporter, lighting and
  ground plane in place.
- RViz: same as before, except the camera TF no longer drifts when the UR5 moves.

---

## 8. Open question / decision log

- The two `CMakeLists.txt` edits (1a, 1c) technically fall outside `ur5_endo_flex`
  and modify vendored upstream packages. If the user prefers zero edits to
  dependencies, the alternative is a "shim" approach: drop the same
  `model.config` files in `src/ur5_endo_flex/share_gz_models/{ur_description,ur5_endoscope}/`
  with symlinks back to the real meshes, and set `GZ_SIM_RESOURCE_PATH` in
  the launch file only. Slightly larger diff in `ur5_endo_flex/`, but
  zero edits to its dependencies. **Default plan above uses the package-CMakeLists
  edit** for smaller absolute footprint.
- Once Issue 2 is fixed (rigid flex), Issue 1 may be revisited if the user later
  wants true continuum bending: re-introduce revolute/prismatic joints inside the
  same macro, add `position` state + command interfaces to the existing
  `<ros2_control>` block, and add a `JointGroupPositionController` for the flex
  chain in a new controller YAML. Out of scope for this round per the user's
  "I care more about geometry" guidance.

---

## 9. Reference

- Test run log: `/tmp/ur5_logs/launch.log` (524 lines)
- `ros_gz_sim create` confirmed entity creation; mesh resolution failures are
  the only render-time problem.
- All error categories are reproducible by re-running section 7 above on the
  unpatched tree.
