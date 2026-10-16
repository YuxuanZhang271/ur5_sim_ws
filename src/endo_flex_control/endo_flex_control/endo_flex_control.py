#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tkinter as tk
from pynput import keyboard


N          = 8
SEG_LEN    = 0.008
ARC_MIN    = 0.030
ARC_MAX    = 0.050
MAX_BEND   = math.pi / 2

TENDON_R   = 0.004                                   # tendon offset (m)
TENDONS    = [(0, 'R'), (90, 'G'), (180, 'B'), (270, 'Y')]


class EndoscopeTeleop(Node):
    def __init__(self):
        super().__init__('endoscope_teleop')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # ---------- state ----------
        self.base_yaw = 0.0
        self.arc_len  = ARC_MAX
        self.ext_yaw  = 0.0
        self.mode     = 1

        # curvature κ = 1/R  (1/m).  κ = 0  → perfectly straight.
        self.curvature = 1.0 / 0.050          # start at R = 5 cm

        # ---------- steps ----------
        self.curvature_step = 2.0             # Δκ per ↑/↓ press
        self.yaw_step       = 0.05
        self.len_step       = 0.001
        self.ext_yaw_step   = 0.05

        # ---------- GUI ----------
        self.root = tk.Tk()
        self.root.title("Endoscope Control")
        self.root.geometry("760x520")

        self.mode_lbl = tk.Label(self.root, font=("Arial", 12, "bold"))
        self.mode_lbl.pack(pady=6)

        self.labels = []
        for _ in range(8):
            lbl = tk.Label(self.root, font=("Consolas", 11))
            lbl.pack(pady=2)
            self.labels.append(lbl)

        tk.Label(self.root,
                 text=("↑↓ bend (↑ straighter, κ→0 = straight; ↓ more curved)      "
                       "←→ yaw  (free: base only | track: self-rotate, "
                       "Δext_yaw = -Δbase_yaw)      I/O extend\n"
                       "[ ] ext_yaw (free mode only)      M toggle mode"),
                 font=("Arial", 9)).pack(pady=8)

        keyboard.Listener(on_press=self.on_press).start()
        self.root.after(80, self.update_gui)
        self.publish_joint_states()

    # ------------------------------------------------------------------
    def compute_derived(self):
        eff   = max(1e-6, self.arc_len - SEG_LEN)
        kappa = max(0.0, self.curvature)

        # θ = L · κ   (straight when κ = 0)
        theta_total = eff * kappa
        if theta_total > MAX_BEND:
            theta_total = MAX_BEND

        theta_seg = theta_total / (N - 1)
        d_seg     = eff / (N - 1)

        # Radius for display only; ∞ when perfectly straight.
        R_disp = (1.0 / kappa) if kappa > 1e-9 else float('inf')

        return self.ext_yaw, theta_seg, d_seg, theta_total, R_disp

    # ------------------------------------------------------------------
    def compute_tendon_lengths(self, theta_total):
        """Return {name: (L_m, dL_m)} for the 4 tendons.
        Tendon at world angle φ, tube bending toward α = base_yaw+ext_yaw:
            L(φ) = arc_len − TENDON_R · cos(φ − α) · θ_total
        """
        alpha = self.base_yaw + self.ext_yaw
        out = {}
        for phi_deg, name in TENDONS:
            phi = math.radians(phi_deg)
            dL  = -TENDON_R * math.cos(phi - alpha) * theta_total
            out[name] = (self.arc_len + dL, dL)
        return out

    # ------------------------------------------------------------------
    def publish_joint_states(self):
        ext_yaw, theta_seg, d_seg, theta_total, _ = self.compute_derived()

        names, positions = [], []
        names.append('base_yaw'); positions.append(self.base_yaw)

        for i in range(1, N):
            names.append(f'ext_yaw_{i}');      positions.append( ext_yaw)
            names.append(f'ext_pitch_{i}');    positions.append( theta_seg)
            names.append(f'ext_yaw_back_{i}'); positions.append(-ext_yaw)
            names.append(f'extension_{i}');    positions.append( d_seg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = names
        msg.position = positions
        self.pub.publish(msg)

    # ------------------------------------------------------------------
    def on_press(self, key):
        ch = getattr(key, 'char', None)
        try:
            if key == keyboard.Key.up:
                # straighter:  κ → 0
                self.curvature -= self.curvature_step
            elif key == keyboard.Key.down:
                # more curved
                self.curvature += self.curvature_step

            elif key == keyboard.Key.left:
                self.base_yaw += self.yaw_step
                if self.mode == 2:
                    self.ext_yaw -= self.yaw_step
            elif key == keyboard.Key.right:
                self.base_yaw -= self.yaw_step
                if self.mode == 2:
                    self.ext_yaw += self.yaw_step

            elif ch == 'i':
                self.arc_len = min(ARC_MAX, self.arc_len + self.len_step)
            elif ch == 'o':
                self.arc_len = max(ARC_MIN, self.arc_len - self.len_step)

            elif ch == '[':
                if self.mode == 1:
                    self.ext_yaw -= self.ext_yaw_step
            elif ch == ']':
                if self.mode == 1:
                    self.ext_yaw += self.ext_yaw_step

            elif ch == 'm':
                self.mode = 2 if self.mode == 1 else 1

            else:
                return

            # κ ≥ 0 (κ = 0  → perfectly straight).
            # upper bound keeps θ_total = eff·κ ≤ MAX_BEND.
            eff = max(1e-6, self.arc_len - SEG_LEN)
            kappa_max = MAX_BEND / eff
            self.curvature = max(0.0, min(kappa_max, self.curvature))

            self.publish_joint_states()
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    def update_gui(self):
        ext_yaw, theta_seg, d_seg, theta_total, R_disp = self.compute_derived()

        if self.mode == 1:
            mode_txt = "Mode 1  free (←/→ rotate whole tube; ext_yaw fixed)"
        else:
            mode_txt = ("Mode 2  self-rotate (←/→ twist tube; tip preserved, "
                        "Δext_yaw = -Δbase_yaw)")
        self.mode_lbl.config(text=mode_txt)

        if math.isinf(R_disp):
            r_str = "   inf"
            k_str = f"{self.curvature:5.2f}"
        else:
            r_str = f"{R_disp*100:6.2f}"
            k_str = f"{self.curvature:5.2f}"

        self.labels[0].config(
            text=f"base_yaw    = {math.degrees(self.base_yaw):+7.2f} deg")
        self.labels[1].config(
            text=f"arc L       = {self.arc_len*100:6.2f} cm    "
                 f"R = {r_str} cm    κ = {k_str} 1/m")
        self.labels[2].config(
            text=f"theta_tot   = {math.degrees(theta_total):+7.2f} deg   "
                 f"theta_seg  = {math.degrees(theta_seg):+6.2f} deg")
        self.labels[3].config(
            text=f"ext_yaw/seg = {math.degrees(ext_yaw):+7.2f} deg   "
                 f"extend_seg = {d_seg*1000:.3f} mm")

        alpha = self.base_yaw + self.ext_yaw
        tip_x = math.sin(theta_total) * math.cos(alpha)
        tip_y = math.sin(theta_total) * math.sin(alpha)
        tip_z = math.cos(theta_total)

        self.labels[4].config(
            text=f"tip yaw     = {math.degrees(alpha):+7.2f} deg   "
                 f"(~const in mode 2)")
        self.labels[5].config(
            text=f"tip dir    ~ ({tip_x:+.3f}, {tip_y:+.3f}, {tip_z:+.3f})")

        # ---------- tendon lengths ----------
        tl = self.compute_tendon_lengths(theta_total)
        L_row = "   ".join(f"{n}={tl[n][0]*1000:6.2f}" for _, n in TENDONS)
        d_row = "   ".join(f"{n}={tl[n][1]*1000:+6.2f}" for _, n in TENDONS)
        self.labels[6].config(text=f"tendon L  [mm] :  {L_row}")
        self.labels[7].config(text=f"tendon ΔL [mm] :  {d_row}")

        self.root.after(80, self.update_gui)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    teleop = EndoscopeTeleop()
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()