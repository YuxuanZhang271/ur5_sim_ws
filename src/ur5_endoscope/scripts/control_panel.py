#!/usr/bin/env python3
"""
Tkinter GUI for the UR5 + endoscope control panel.

Run this file directly:

    python3 control_panel.py

It imports `endoscope_node.ControlPanel` and `endoscope_kinematics`
from the same directory.

Modes
-----
* Manual joint control : type a value into the input box, press Enter
                         (or click Send) to command the arm.
* RCM mode             : arrow keys / tilt buttons pivot the tube
                         about the fixed RCM point in world-frame
                         directions.

For each joint the panel shows two values side by side:
    * Input  [deg] : what the user has typed (sent on Send).
    * Robot  [deg] : live read-back from /joint_states (read-only).

The "Keep wrist_1 pointing out" checkbox auto-solves wrist_1 so that

    shoulder_lift + elbow + wrist_1 = -270 deg

which keeps the wrist_1 joint axis pointing out from the supporter and
base regardless of the other joints.
"""

import threading
import numpy as np
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.executors import SingleThreadedExecutor

from endoscope_kinematics import JOINT_NAMES, HOME_DEG
from endoscope_node import ControlPanel


class ControlPanelGUI:
    def __init__(self, node: ControlPanel):
        self.node = node
        self.root = tk.Tk()
        self.root.title('UR5 + Endoscope control panel')
        self.root.geometry('720x1000')

        self.entry_vars = []
        self.entry_widgets = []
        self.state_vars = []    # live robot joint angles, read-only
        self._syncing = False
        # Widgets that own an input value — used to suppress the
        # auto-sync of the live state label while the user is typing
        # into an entry.  Declared early so _refresh_ui can use it
        # on the first tick.
        self._own_input_widgets = set()

        self.keep_out_var = tk.BooleanVar(value=False)

        self._build_widgets()
        self._bind_keys()

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

        self._refresh_ui()

    # ------------------------------------------------------------------
    def _build_widgets(self):
        frame = ttk.LabelFrame(self.root,
                               text='Manual joint control (degrees)')
        frame.pack(fill='x', padx=10, pady=8)

        # --- Mode checkboxes ----------------------------------------
        mode_row = ttk.Frame(frame)
        mode_row.pack(fill='x', padx=6, pady=(4, 2))
        ttk.Checkbutton(
            mode_row,
            text='Keep wrist_1 pointing out (auto-solve)',
            variable=self.keep_out_var,
            command=self._on_keep_out_toggle
        ).pack(side='left')

        hdr = ttk.Frame(frame)
        hdr.pack(fill='x', padx=6, pady=(2, 4))
        ttk.Label(hdr, text='Joint', width=20).pack(side='left')
        ttk.Label(hdr, text='Input [deg]', width=12, anchor='w').pack(
            side='left', padx=(0, 12))
        ttk.Label(hdr, text='Robot [deg]', width=12, anchor='e').pack(
            side='left')

        for i, name in enumerate(JOINT_NAMES):
            row = ttk.Frame(frame)
            row.pack(fill='x', padx=6, pady=2)
            ttk.Label(row, text=name, width=20).pack(side='left')

            evar = tk.StringVar(value=f'{HOME_DEG[i]:.2f}')
            e = ttk.Entry(row, textvariable=evar, width=10, justify='right')
            e.pack(side='left', padx=(0, 12))
            e.bind('<Return>', lambda ev, idx=i: self._on_entry_commit(idx))
            e.bind('<KP_Enter>', lambda ev, idx=i: self._on_entry_commit(idx))
            e.bind('<FocusOut>', lambda ev, idx=i: self._on_entry_commit(idx))
            evar.trace_add('write',
                           lambda *a, idx=i: self._on_entry_live(idx))

            # Live read-back of the actual robot joint angle.
            svar = tk.StringVar(value='-.--')
            ttk.Label(row, textvariable=svar, width=10,
                      anchor='e', foreground='gray').pack(side='left')

            self.entry_vars.append(evar)
            self.entry_widgets.append(e)
            self.state_vars.append(svar)

        # Cache the set of widgets whose value the user is currently
        # editing, so _refresh_ui can leave them alone.
        self._own_input_widgets = set(self.entry_widgets)

        row = ttk.Frame(frame)
        row.pack(fill='x', padx=6, pady=6)
        ttk.Button(row, text='Send',
                   command=self._send_current).pack(side='left')
        ttk.Button(row, text='Zero',
                   command=self._zero_all).pack(side='left', padx=4)
        ttk.Button(row, text='Home',
                   command=self._home_all).pack(side='left', padx=4)

        frame = ttk.LabelFrame(self.root, text='Keyboard linear control')
        frame.pack(fill='x', padx=10, pady=8)
        ttk.Label(frame, justify='left',
                  text=("Focus this window and press:\n"
                        "   'i'  ->  go IN  (advance along endoscope +Z)\n"
                        "   'o'  ->  go OUT (retract)\n"
                        "   Arrow keys -> TILT around RCM (requires RCM ON).\n"
                        "   Directions are in the WORLD frame:\n"
                        "      Up / Down    : pitch the tip toward +Z / -Z\n"
                        "      Left / Right : yaw the tip toward +Y / -Y\n"
                        "   Tilt is rejected if it would push the tube\n"
                        "   more than 20 deg away from the RCM axis\n"
                        "   captured when RCM was enabled.")
                  ).pack(anchor='w', padx=8, pady=4)

        row = ttk.Frame(frame)
        row.pack(fill='x', padx=8, pady=4)
        ttk.Label(row, text='Linear step [mm]:').pack(side='left')
        self.step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(row, textvariable=self.step_var, width=8).pack(
            side='left', padx=4)
        ttk.Button(row, text='IN (i)',
                   command=lambda: self._on_step(+1)).pack(
                       side='left', padx=4)
        ttk.Button(row, text='OUT (o)',
                   command=lambda: self._on_step(-1)).pack(side='left')

        row = ttk.Frame(frame)
        row.pack(fill='x', padx=8, pady=4)
        ttk.Label(row, text='Tilt step [deg]:').pack(side='left')
        self.tilt_step_var = tk.DoubleVar(value=5.0)
        ttk.Entry(row, textvariable=self.tilt_step_var, width=8).pack(
            side='left', padx=4)

        row = ttk.Frame(frame)
        row.pack(fill='x', padx=8, pady=4)
        ttk.Button(row, text='\u25B2  UP',
                   command=lambda: self._on_tilt(0.0, +1.0)).pack(side='left')
        ttk.Button(row, text='\u25C0  LEFT',
                   command=lambda: self._on_tilt(-1.0, 0.0)).pack(
                       side='left', padx=4)
        ttk.Button(row, text='RIGHT  \u25B6',
                   command=lambda: self._on_tilt(+1.0, 0.0)).pack(side='left')
        ttk.Button(row, text='\u25BC  DOWN',
                   command=lambda: self._on_tilt(0.0, -1.0)).pack(
                       side='left', padx=4)
        ttk.Label(frame,
                  text=('(Tilt buttons / arrow keys only act when RCM is ON. '
                        'World-frame directions.)'),
                  foreground='gray').pack(anchor='w', padx=8, pady=(0, 4))

        frame = ttk.LabelFrame(self.root,
                               text='RCM (remote centre of motion)')
        frame.pack(fill='x', padx=10, pady=8)
        self.rcm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text='RCM enabled', variable=self.rcm_var,
                        command=self._on_rcm_toggle).pack(
                            anchor='w', padx=8, pady=4)
        self.rcm_status = ttk.Label(frame, text='RCM: OFF')
        self.rcm_status.pack(anchor='w', padx=8)

        frame = ttk.LabelFrame(self.root, text='Visualization')
        frame.pack(fill='x', padx=10, pady=8)
        self.markers_var = tk.BooleanVar(value=self.node.markers_enabled)
        ttk.Checkbutton(frame,
                        text=('Publish RViz markers '
                              f'(topic: {self.node.marker_topic})'),
                        variable=self.markers_var,
                        command=self._on_markers_toggle
                        ).pack(anchor='w', padx=8, pady=4)
        ttk.Label(
            frame,
            justify='left',
            text=('IMPORTANT: these markers are only visible in RViz2.\n'
                  'In RViz2:\n'
                  '  - Fixed Frame = base_link (or world)\n'
                  '  - Add -> By topic -> '
                  f'{self.node.marker_topic} -> MarkerArray\n'
                  '  - NOT "Marker" -- pick "MarkerArray".')
            ).pack(anchor='w', padx=8, pady=4)

        self.status = ttk.Label(self.root, text='Ready.',
                                relief='sunken', anchor='w')
        self.status.pack(fill='x', side='bottom')

    def _bind_keys(self):
        self.root.bind('<KeyPress-i>', lambda e: self._on_step(+1))
        self.root.bind('<KeyPress-o>', lambda e: self._on_step(-1))
        self.root.bind('<KeyPress-I>', lambda e: self._on_step(+1))
        self.root.bind('<KeyPress-O>', lambda e: self._on_step(-1))
        self.root.bind('<KeyPress-Up>',
                       lambda e: self._on_tilt(0.0, +1.0))
        self.root.bind('<KeyPress-Down>',
                       lambda e: self._on_tilt(0.0, -1.0))
        self.root.bind('<KeyPress-Left>',
                       lambda e: self._on_tilt(-1.0, 0.0))
        self.root.bind('<KeyPress-Right>',
                       lambda e: self._on_tilt(+1.0, 0.0))
        self.root.focus_force()

    # ------------------------------------------------------------------
    # Keep-out helper (wrist_1 auto-solve)
    # ------------------------------------------------------------------
    def _keep_out_on(self):
        """Whether wrist_1 is currently auto-solved by keep-out mode."""
        return self.keep_out_var.get()

    # ------------------------------------------------------------------
    # Entry plumbing
    # ------------------------------------------------------------------
    def _on_entry_live(self, idx):
        if self._syncing:
            return
        txt = self.entry_vars[idx].get().strip()
        if txt in ('', '+', '-', '.', '+.', '-.'):
            return
        try:
            float(txt)
        except ValueError:
            return
        if idx != 3 and self._keep_out_on():
            self._recompute_wrist1()

    def _on_entry_commit(self, idx):
        txt = self.entry_vars[idx].get().strip()
        try:
            val = float(txt)
        except ValueError:
            self._set_status(f'Joint {idx + 1}: invalid entry, ignored.')
            return
        clamped = max(-180.0, min(180.0, val))
        self._syncing = True
        try:
            self.entry_vars[idx].set(f'{clamped:.2f}')
        finally:
            self._syncing = False
        self._set_status(f'Joint {idx + 1} set to {clamped:+.2f} deg')
        if idx != 3 and self._keep_out_on():
            self._recompute_wrist1()

    def _read_all_deg(self):
        out = np.zeros(6)
        for i in range(6):
            txt = self.entry_vars[i].get().strip()
            try:
                out[i] = float(txt)
            except ValueError:
                pass  # already 0
        return out

    # ------------------------------------------------------------------
    # Keep-out helper (wrist_1 auto-solve)
    # ------------------------------------------------------------------
    def _recompute_wrist1(self):
        if not self.node.kin.ready:
            return
        q_deg = self._read_all_deg()
        q_rad = np.deg2rad(q_deg)
        q_solved = self.node.kin.solve_wrist1_for_out(q_rad)
        w1_deg = float(np.rad2deg(q_solved[3]))
        while w1_deg > 180.0:
            w1_deg -= 360.0
        while w1_deg < -180.0:
            w1_deg += 360.0
        self._syncing = True
        try:
            self.entry_vars[3].set(f'{w1_deg:.2f}')
        finally:
            self._syncing = False

    def _on_keep_out_toggle(self):
        on = bool(self._keep_out_on())
        w = self.entry_widgets[3]
        try:
            if on:
                w.state(['disabled'])
            else:
                w.state(['!disabled'])
        except tk.TclError:
            w.configure(state=('disabled' if on else 'normal'))
        if on:
            self._recompute_wrist1()
            self._set_status(
                'Keep-out ON: wrist_1 auto-solved so '
                'shoulder_lift + elbow + wrist_1 = -270 deg.')
        else:
            self._set_status('Keep-out OFF: wrist_1 controlled manually.')

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------
    def _send_current(self):
        q_deg = self._read_all_deg()
        q_rad = np.deg2rad(q_deg)
        if self._keep_out_on() and self.node.kin.ready:
            q_rad = self.node.kin.solve_wrist1_for_out(q_rad)
            w1_deg = float(np.rad2deg(q_rad[3]))
            while w1_deg > 180.0:
                w1_deg -= 360.0
            while w1_deg < -180.0:
                w1_deg += 360.0
            self._syncing = True
            try:
                self.entry_vars[3].set(f'{w1_deg:.2f}')
            finally:
                self._syncing = False
        self.node.set_joint_positions(q_rad)
        self._set_status(
            f'Sent joints (deg) '
            f'{np.round(np.rad2deg(q_rad), 2).tolist()}')

    def _zero_all(self):
        self._syncing = True
        try:
            for i in range(6):
                if i == 3 and self._keep_out_on():
                    continue
                self.entry_vars[i].set('0.00')
        finally:
            self._syncing = False
        if self._keep_out_on():
            self._recompute_wrist1()
        self._send_current()

    def _home_all(self):
        self._syncing = True
        try:
            for i in range(6):
                if i == 3 and self._keep_out_on():
                    continue
                self.entry_vars[i].set(f'{HOME_DEG[i]:.2f}')
        finally:
            self._syncing = False
        if self._keep_out_on():
            self._recompute_wrist1()
        self._send_current()

    def _on_step(self, sign):
        w = self.root.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            return
        try:
            step_m = float(self.step_var.get()) / 1000.0
        except Exception:
            step_m = 0.005
        ok = self.node.step_along_axis(-sign * step_m)
        self._set_status(
            f'{"IN" if sign > 0 else "OUT"} '
            f'{sign * step_m * 1000:+.1f} mm '
            + ('OK' if ok else 'FAILED'))

    def _on_tilt(self, right_sign, up_sign):
        w = self.root.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            return None
        try:
            step_deg = float(self.tilt_step_var.get())
        except Exception:
            step_deg = 5.0
        ok = self.node.tilt_with_rcm(right_sign * step_deg,
                                     up_sign * step_deg)
        parts = []
        if right_sign > 0:
            parts.append('RIGHT')
        elif right_sign < 0:
            parts.append('LEFT')
        if up_sign > 0:
            parts.append('UP')
        elif up_sign < 0:
            parts.append('DOWN')
        self._set_status(
            f'TILT (world) {"+".join(parts)} {step_deg:.1f} deg '
            + ('OK' if ok else 'FAILED (RCM off / cone limit reached)'))
        return 'break'

    def _on_rcm_toggle(self):
        if self.rcm_var.get():
            ok = self.node.enable_rcm()
            if not ok:
                self.rcm_var.set(False)
                self._set_status('RCM enable failed')
                return
            self._set_status('RCM enabled')
        else:
            self.node.disable_rcm()
            self._set_status('RCM disabled')

    def _on_markers_toggle(self):
        self.node.markers_enabled = bool(self.markers_var.get())
        self._set_status(
            f'Markers {"ON" if self.node.markers_enabled else "OFF"}')

    # ------------------------------------------------------------------
    # Periodic UI refresh
    # ------------------------------------------------------------------
    def _refresh_ui(self):
        with self.node.rcm_lock:
            on = self.node.rcm_enabled
            p = None if self.node.rcm_world_point is None \
                else self.node.rcm_world_point.copy()
        if on and p is not None:
            self.rcm_status.config(
                text=('RCM: ON   point=('
                      f'{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m'))
        else:
            self.rcm_status.config(
                text='RCM: OFF (point moves with endoscope)')

        if self.node.joint_received:
            # Live read-back of /joint_states -> state labels.  We
            # never touch entry_vars here; the user's typed value
            # stays put until they hit Send.
            with self.node.joint_lock:
                q = self.node.joint_positions.copy()
            q_deg = np.rad2deg(q)
            for i in range(6):
                self.state_vars[i].set(f'{q_deg[i]:+.2f}')

        self.root.after(200, self._refresh_ui)

    def _set_status(self, s):
        self.status.config(text=s)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Tear-down order matters.  The executor must be stopped and
            # the spin thread must finish before the node is destroyed,
            # otherwise rclpy raises
            #   "terminate called without an active exception"
            # while the spin thread is still touching the C++ node.
            try:
                self.executor.remove_node(self.node)
            except Exception:
                pass
            self.executor.shutdown()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=3.0)
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ControlPanel()
    gui = ControlPanelGUI(node)
    gui.run()


if __name__ == '__main__':
    main()