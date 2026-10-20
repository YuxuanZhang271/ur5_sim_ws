#!/usr/bin/env python3
"""
control_panel_gui.py — Compact Tkinter control panel for ur5_endo_flex.

Compact Layout (~580x530 default, expandable):
    Row 0:  Top toolbar (HOME, RESET, BIRD-EYE, PULL OUT)
    Row 1:  Arm state (3x2 joint grid + RCM status) | Arm controls (mode toggle + 2x4 D-pad)
    Row 2:  Flex state (3x2 param grid + tendons)   | Flex controls (mode toggle + 2x4 WASD + R/F)
    Row 3:  Target tracking (4 target buttons + STOP + live status)
    Row 4:  Permanent Keyboard Shortcuts panel at the bottom
"""

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

import tkinter as tk
from tkinter import ttk

from ur5_endo_flex_node import (
    Ur5EndoFlexNode, FLEX_TENDONS, TRACK_TARGETS)


class FlexGridGUI:
    def __init__(self, node: Ur5EndoFlexNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('ur5_endo_flex — control panel')
        self.root.geometry('580x530')
        self.root.minsize(520, 490)

        # 2 columns: State (col 0) | Controls (col 1)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)

        # Row 0 (toolbar), Row 3 (tracking) and Row 4 (keytips) take minimum
        # height; Rows 1 (Arm) and 2 (Flex) share equal vertical space.
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1, uniform='section_row')
        self.root.rowconfigure(2, weight=1, uniform='section_row')
        self.root.rowconfigure(3, weight=0)
        self.root.rowconfigure(4, weight=0)

        self._button_pressed = set()

        self._build_toolbar()
        self._build_arm_row()
        self._build_flex_row()
        self._build_tracking_row()
        self._build_keytips()

        # Root-level ButtonRelease clears orphaned presses.
        self.root.bind('<ButtonRelease-1>',
                       lambda e: self._clear_all_button_keys())

        # Spin node in a background thread.
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

        self._refresh()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _labeled_frame(self, parent, title):
        return ttk.LabelFrame(parent, text=title, padding=(6, 4))

    def _bind_press_hold(self, widget, key):
        widget.bind('<ButtonPress-1>',
                    lambda e, k=key: self._on_button_press(k))
        widget.bind('<ButtonRelease-1>',
                    lambda e, k=key: self._on_button_release(k))

    def _on_button_press(self, key):
        self._button_pressed.add(key)
        self.node.set_key(key, True)

    def _on_button_release(self, key):
        self._button_pressed.discard(key)
        self.node.set_key(key, False)

    def _clear_all_button_keys(self):
        for key in list(self._button_pressed):
            self._button_pressed.discard(key)
            self.node.set_key(key, False)

    # ------------------------------------------------------------------
    # Row 0: Compact Top Toolbar (Semi-auto commands)
    # ------------------------------------------------------------------
    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(4, 4, 4, 0))
        bar.grid(row=0, column=0, columnspan=2, sticky='ew')
        for c in range(4):
            bar.columnconfigure(c, weight=1)

        ttk.Button(bar, text='HOME (Ctrl+H)',
                   command=self.node.cmd_home).grid(
            row=0, column=0, padx=2, sticky='ew')
        ttk.Button(bar, text='RESET (Ctrl+R)',
                   command=self.node.cmd_reset).grid(
            row=0, column=1, padx=2, sticky='ew')
        ttk.Button(bar, text='BIRD-EYE (Ctrl+B)',
                   command=self.node.cmd_bird_eye_view).grid(
            row=0, column=2, padx=2, sticky='ew')
        ttk.Button(bar, text='PULL OUT (Ctrl+P)',
                   command=self.node.cmd_pull_out).grid(
            row=0, column=3, padx=2, sticky='ew')

    # ------------------------------------------------------------------
    # Row 1: Arm section (State in 3x2 grid | Controls in 2x4 grid)
    # ------------------------------------------------------------------
    def _build_arm_row(self):
        # ---- Col 0: Arm state ----
        f1 = self._labeled_frame(self.root, 'Arm state')
        f1.grid(row=1, column=0, sticky='nsew', padx=(4, 2), pady=3)
        f1.columnconfigure(0, weight=1)
        f1.columnconfigure(1, weight=1)

        hdr = ttk.Frame(f1)
        hdr.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 3))
        self.arm_mode_var = tk.StringVar(value='Mode: linear')
        ttk.Label(hdr, textvariable=self.arm_mode_var,
                  font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        self.arm_rcm_var = tk.StringVar(value='RCM: OFF')
        ttk.Label(hdr, textvariable=self.arm_rcm_var,
                  font=('Consolas', 8), foreground='gray').pack(side='right')

        # 3x2 compact joint grid
        self.arm_joint_vars = []
        short_joints = [
            'J1 pan', 'J2 lift', 'J3 elbow',
            'J4 wr1', 'J5 wr2',  'J6 wr3',
        ]
        jg = ttk.Frame(f1)
        jg.grid(row=1, column=0, columnspan=2, sticky='nsew')
        for c in range(2):
            jg.columnconfigure(c, weight=1)
        for idx, label in enumerate(short_joints):
            r, c = divmod(idx, 2)
            cell = ttk.Frame(jg)
            cell.grid(row=r, column=c, sticky='ew', padx=3, pady=1)
            ttk.Label(cell, text=label, width=8,
                      font=('TkDefaultFont', 8)).pack(side='left')
            v = tk.StringVar(value='-.-\u00b0')
            ttk.Label(cell, textvariable=v, width=8, anchor='e',
                      font=('Consolas', 9)).pack(side='right')
            self.arm_joint_vars.append(v)

        # ---- Col 1: Arm controls ----
        f2 = self._labeled_frame(self.root, 'Arm controls')
        f2.grid(row=1, column=1, sticky='nsew', padx=(2, 4), pady=3)
        f2.columnconfigure(0, weight=1)
        f2.rowconfigure(1, weight=1)

        self.arm_mode_btn_var = tk.StringVar(value='Mode: linear (Ctrl+M)')
        ttk.Button(f2, textvariable=self.arm_mode_btn_var,
                   command=self._on_arm_mode_toggle).grid(
            row=0, column=0, sticky='ew', pady=(0, 4))

        gf = ttk.Frame(f2)
        gf.grid(row=1, column=0, sticky='nsew')
        for c in range(4):
            gf.columnconfigure(c, weight=1, uniform='arm_btn')
        for r in range(2):
            gf.rowconfigure(r, weight=1, uniform='arm_row')

        btn_specs = [
            ('\u2191',     0, 1, 'arm_up'),
            ('\u2190',     1, 0, 'arm_left'),
            ('\u2193',     1, 1, 'arm_down'),
            ('\u2192',     1, 2, 'arm_right'),
            ('Fwd [',      0, 3, 'arm_fwd'),
            ('Bwd ]',      1, 3, 'arm_bwd'),
        ]
        for text, r, c, key in btn_specs:
            b = ttk.Button(gf, text=text, width=5)
            b.grid(row=r, column=c, padx=2, pady=2, sticky='nsew')
            self._bind_press_hold(b, key)

    # ------------------------------------------------------------------
    # Row 2: Flex section (State in 3x2 grid + tendons | Controls in 2x4 grid)
    # ------------------------------------------------------------------
    def _build_flex_row(self):
        # ---- Col 0: Flex state ----
        f1 = self._labeled_frame(self.root, 'Flex state')
        f1.grid(row=2, column=0, sticky='nsew', padx=(4, 2), pady=3)
        f1.columnconfigure(0, weight=1)

        hdr = ttk.Frame(f1)
        hdr.grid(row=0, column=0, sticky='ew', pady=(0, 3))
        self.flex_mode_var = tk.StringVar(value='Mode: fixed')
        ttk.Label(hdr, textvariable=self.flex_mode_var,
                  font=('TkDefaultFont', 10, 'bold')).pack(side='left')

        # 3x2 compact flex parameter grid
        self.flex_param_vars = {}
        param_specs = [
            ('base_yaw',    'base_yaw'),
            ('ext_yaw',     'ext_yaw'),
            ('arc_len',     'arc_len'),
            ('curvature',   'curv \u03ba'),
            ('theta_total', '\u03b8_total'),
            ('theta_seg',   '\u03b8_seg'),
        ]
        pg = ttk.Frame(f1)
        pg.grid(row=1, column=0, sticky='ew')
        for c in range(2):
            pg.columnconfigure(c, weight=1)
        for idx, (key, short_lbl) in enumerate(param_specs):
            r, c = divmod(idx, 2)
            cell = ttk.Frame(pg)
            cell.grid(row=r, column=c, sticky='ew', padx=3, pady=1)
            ttk.Label(cell, text=short_lbl, width=8,
                      font=('TkDefaultFont', 8)).pack(side='left')
            v = tk.StringVar(value='-.--')
            ttk.Label(cell, textvariable=v, width=9, anchor='e',
                      font=('Consolas', 9)).pack(side='right')
            self.flex_param_vars[key] = v

        # Compact 2-line tendon readout
        tf = ttk.Frame(f1)
        tf.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        self.tendon_L_var = tk.StringVar(value='L : -')
        self.tendon_dL_var = tk.StringVar(value='\u0394L: -')
        ttk.Label(tf, textvariable=self.tendon_L_var,
                  font=('Consolas', 8)).pack(anchor='w')
        ttk.Label(tf, textvariable=self.tendon_dL_var,
                  font=('Consolas', 8)).pack(anchor='w')

        # ---- Col 1: Flex controls (2x4 grid: WASD + R/F) ----
        f2 = self._labeled_frame(self.root, 'Flex controls')
        f2.grid(row=2, column=1, sticky='nsew', padx=(2, 4), pady=3)
        f2.columnconfigure(0, weight=1)
        f2.rowconfigure(1, weight=1)

        self.flex_mode_btn_var = tk.StringVar(value='Mode: fixed (Ctrl+N)')
        ttk.Button(f2, textvariable=self.flex_mode_btn_var,
                   command=self._on_flex_mode_toggle).grid(
            row=0, column=0, sticky='ew', pady=(0, 4))

        gf = ttk.Frame(f2)
        gf.grid(row=1, column=0, sticky='nsew')
        for c in range(4):
            gf.columnconfigure(c, weight=1, uniform='flex_btn')
        for r in range(2):
            gf.rowconfigure(r, weight=1, uniform='flex_row')

        for text, r, c, key in [
            ('W', 0, 1, 'flex_w'),
            ('A', 1, 0, 'flex_a'),
            ('S', 1, 1, 'flex_s'),
            ('D', 1, 2, 'flex_d'),
            ('R', 0, 3, 'flex_r'),
            ('F', 1, 3, 'flex_f'),
        ]:
            b = ttk.Button(gf, text=text, width=5)
            b.grid(row=r, column=c, padx=2, pady=2, sticky='nsew')
            self._bind_press_hold(b, key)

    # ------------------------------------------------------------------
    # Row 3: Target tracking (4 target buttons + STOP + live status)
    # ------------------------------------------------------------------
    def _build_tracking_row(self):
        f = self._labeled_frame(self.root, 'Target tracking')
        f.grid(row=3, column=0, columnspan=2, sticky='ew', padx=4, pady=3)

        # [status label (wider)] [4 target buttons] [STOP]
        f.columnconfigure(0, weight=2)
        for c in range(1, len(TRACK_TARGETS) + 2):
            f.columnconfigure(c, weight=1, uniform='track_btn')

        self.tracking_status_var = tk.StringVar(value='Tracking: idle')
        ttk.Label(f, textvariable=self.tracking_status_var,
                  font=('Consolas', 8)).grid(
            row=0, column=0, sticky='w', padx=(2, 8))

        self._track_buttons = {}
        for i, (key, label, _rgba) in enumerate(TRACK_TARGETS):
            b = ttk.Button(
                f, text=label,
                command=lambda k=key: self.node.start_tracking(k))
            b.grid(row=0, column=1 + i, padx=2, sticky='ew')
            self._track_buttons[key] = (b, label)

        ttk.Button(f, text='\u25a0 STOP',
                   command=self._on_stop_tracking).grid(
            row=0, column=len(TRACK_TARGETS) + 1, padx=2, sticky='ew')

    # ------------------------------------------------------------------
    # Row 4: Permanent Keyboard Shortcuts Panel at Bottom
    # ------------------------------------------------------------------
    def _build_keytips(self):
        self.keytips_frame = self._labeled_frame(self.root, 'Keyboard Shortcuts')
        self.keytips_frame.grid(
            row=4, column=0, columnspan=2, sticky='ew', padx=4, pady=(0, 4))
        self.keytips_frame.columnconfigure(0, weight=1)
        self.keytips_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.keytips_frame, justify='left', font=('Consolas', 8),
            text=(
                "Arm (Linear): \u2191/\u2193 \u00b1Z | \u2190/\u2192 horiz \u22a5 | [/] along axis\n"
                "Arm (RCM)   : \u2191/\u2193 tilt | \u2190/\u2192 yaw | [/] in/out (\u00b120\u00b0)\n"
                "Modes       : Ctrl+M arm | Ctrl+N flex"
            )
        ).grid(row=0, column=0, sticky='nw', padx=(4, 12))

        ttk.Label(
            self.keytips_frame, justify='left', font=('Consolas', 8),
            text=(
                "Flex : W/S straighten/bend | A/D yaw | R/F ext len\n"
                "Auto : Ctrl+H HOME | Ctrl+R RESET | Ctrl+B BIRD-EYE | Ctrl+P PULL OUT\n"
                "Tracking: buttons above (RCM + fixed flex) | any manual key stops"
            )
        ).grid(row=0, column=1, sticky='nw', padx=4)

    # ------------------------------------------------------------------
    # Periodic UI refresh
    # ------------------------------------------------------------------
    def _refresh(self):
        try:
            s = self.node.get_state_dict()
        except Exception as e:
            print('refresh error:', e)
            self.root.after(200, self._refresh)
            return

        # Arm state.
        self.arm_mode_var.set(f'Mode: {s["arm_mode"]}')
        self.arm_mode_btn_var.set(f'Mode: {s["arm_mode"]} (Ctrl+M)')
        if s['joints_deg'] is not None:
            for i, v in enumerate(s['joints_deg']):
                self.arm_joint_vars[i].set(f'{v:+6.1f}\u00b0')
        if s['rcm_enabled'] and s['rcm_point'] is not None:
            p = s['rcm_point']
            self.arm_rcm_var.set(
                f'RCM: ON ({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})')
        else:
            self.arm_rcm_var.set('RCM: OFF')

        # Flex state.
        self.flex_mode_var.set(f'Mode: {s["flex_mode"]}')
        self.flex_mode_btn_var.set(f'Mode: {s["flex_mode"]} (Ctrl+N)')
        self.flex_param_vars['base_yaw'].set(
            f'{s["flex_base_yaw"]:+6.2f}\u00b0')
        self.flex_param_vars['arc_len'].set(
            f'{s["flex_arc_len"] * 100:5.2f} cm')
        self.flex_param_vars['ext_yaw'].set(
            f'{s["flex_ext_yaw"]:+6.2f}\u00b0')
        self.flex_param_vars['curvature'].set(
            f'{s["flex_curvature"]:5.2f}/m')
        self.flex_param_vars['theta_total'].set(
            f'{s["flex_theta_total_deg"]:+6.2f}\u00b0')
        self.flex_param_vars['theta_seg'].set(
            f'{s["flex_theta_seg_deg"]:+6.2f}\u00b0')

        L_row = ' '.join(
            f'{n}:{s["tendons"][n][0]:5.1f}' for _, n in FLEX_TENDONS)
        d_row = ' '.join(
            f'{n}:{s["tendons"][n][1]:+5.1f}' for _, n in FLEX_TENDONS)
        self.tendon_L_var.set(f'L [mm] {L_row}')
        self.tendon_dL_var.set(f'\u0394L[mm] {d_row}')

        # Target tracking.
        tracking = s['tracking']
        zone_labels = {1: 'Flex only', 2: 'Blending', 3: 'Arm takeover'}
        if tracking is not None:
            d = s['tracking_dist_m']
            dist_txt = '--' if d is None else f'{d * 100:.1f}'
            aim = s.get('tracking_aim_deg')
            aim_txt = '--' if aim is None else f'{aim:.1f}'
            zone_txt = zone_labels.get(s.get('tracking_zone'), '--')
            self.tracking_status_var.set(
                f'Tracking: {tracking} \u00b7 {dist_txt} cm \u00b7 '
                f'aim {aim_txt}\u00b0 \u00b7 {zone_txt}'
                + (' \u00b7 locked' if s['tracking_locked'] else ''))
        elif s.get('tracking_refusal'):
            self.tracking_status_var.set(
                f'Tracking: {s["tracking_refusal"]}')
        else:
            self.tracking_status_var.set('Tracking: idle')
        for key, (btn, label) in self._track_buttons.items():
            btn.configure(
                text=('\u25b6 ' + label) if key == tracking else label)

        self.root.after(200, self._refresh)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_arm_mode_toggle(self):
        self.node.toggle_arm_mode()

    def _on_flex_mode_toggle(self):
        self.node.toggle_flex_mode()

    def _on_stop_tracking(self):
        self.node.stop_tracking()

    # ------------------------------------------------------------------
    # Mainloop lifecycle
    # ------------------------------------------------------------------
    def run(self):
        try:
            self.root.mainloop()
        finally:
            try:
                self.executor.remove_node(self.node)
            except Exception:
                pass
            self.executor.shutdown()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = Ur5EndoFlexNode()
    gui = FlexGridGUI(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()