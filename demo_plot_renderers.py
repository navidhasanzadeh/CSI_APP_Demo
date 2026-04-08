"""Figure rendering helpers for demo plots."""

from __future__ import annotations

from math import ceil

import numpy as np


class DemoPlotRenderer:
    def __init__(self, window):
        self.window = window

    def plot_ratio(
        self,
        payload: dict,
        *,
        apply_hampel_phase: bool,
        apply_hampel_magnitude: bool,
    ) -> None:
        fig = self.window.figure
        canvas = self.window.canvas
        series = payload.get("series", [])
        total_pairs = len(series)
        fig.clear()
        if total_pairs == 0:
            canvas.draw_idle()
            return

        grid = fig.add_gridspec(total_pairs, 2, width_ratios=[1, 1], wspace=0.65, hspace=1.05)
        fig_height = max(8, total_pairs * 2.1)
        fig.set_size_inches(12, fig_height)
        canvas.setMinimumHeight(int(fig_height * fig.get_dpi()))

        for row_idx, item in enumerate(series):
            rx_idx = item["rx_idx"]
            tx_num = item["tx_num"]
            tx_den = item["tx_den"]
            x = np.asarray(item["x"], dtype=float)
            ratio_mag = np.asarray(item["ratio_mag"], dtype=float)
            ratio_phase = np.asarray(item["ratio_phase"], dtype=float)
            if apply_hampel_magnitude:
                ratio_mag = self.window._apply_hampel_filter(ratio_mag)
            if apply_hampel_phase:
                ratio_phase = self.window._apply_hampel_filter(ratio_phase)

            ax_mag = fig.add_subplot(grid[row_idx, 0])
            ax_phase = fig.add_subplot(grid[row_idx, 1], sharex=ax_mag)

            if self.window._subplot_visible("csi_ratio_magnitude"):
                ax_mag.plot(x, ratio_mag, color="tab:blue", linewidth=0.9)
                ax_mag.margins(x=0.08, y=0.25)
                self.window._apply_subplot_labels(
                    ax_mag, category="csi_ratio_magnitude",
                    default_title=f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Magnitude",
                    default_xlabel=item["x_label"], default_ylabel="|Ratio|",
                )
                ax_mag.grid(True)
            else:
                ax_mag.set_axis_off()

            if self.window._subplot_visible("csi_ratio_phase"):
                ax_phase.plot(x, ratio_phase, color="tab:green", linewidth=0.9)
                ax_phase.margins(x=0.08, y=0.25)
                self.window._apply_subplot_labels(
                    ax_phase, category="csi_ratio_phase",
                    default_title=f"RX {rx_idx + 1}: TX {tx_num + 1}/TX {tx_den + 1} Phase",
                    default_xlabel=item["x_label"], default_ylabel="Phase (rad)",
                )
                ax_phase.grid(True)
            else:
                ax_phase.set_axis_off()

        fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.07)
        self.window._install_subplot_maximize_buttons(fig, canvas)
        canvas.draw_idle()

    def plot_doppler(self, payload: dict) -> None:
        self.window._plot_doppler_from_payload(payload)

    def plot_dorf(self, payload: dict) -> None:
        self.window._plot_dorf_from_payload(payload)

    def plot_har(self, payload: dict) -> None:
        fig = self.window.har_figure
        canvas = self.window.har_canvas
        fig.clear()
        ax = fig.add_subplot(111)
        if payload.get("status") != "ok":
            msg = payload.get("message", "HAR unavailable until DoRF is ready.")
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            canvas.draw_idle()
            return

        scores = payload.get("scores", {})
        labels = list(scores.keys())
        vals = [float(scores[k]) for k in labels]
        bars = ax.bar(labels, vals, color=["#0ea5e9", "#8b5cf6", "#22c55e"][: len(labels)])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"Predicted Activity: {payload.get('label', 'Unknown')}"
        )
        ax.set_ylabel("Confidence")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        details = f"Model: originalrocket.py + original_rocket_baseline.pkl"
        ax.text(0.5, -0.16, details, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#334155")
        fig.tight_layout()
        canvas.draw_idle()
