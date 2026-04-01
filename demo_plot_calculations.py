"""Computation helpers for demo plotting pipelines."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import sys
from pathlib import Path

import numpy as np

DORF_PATH = Path(__file__).resolve().parent / "DoRF"
if str(DORF_PATH) not in sys.path:
    sys.path.append(str(DORF_PATH))

import doatools.estimation as estimation
from nerfs2 import estimate_velocity_from_radial_old_dtw
from pcap_reader import ProcessPcap
from pcap_reader_ui import _read_ubilocate_csi


class DemoPlotCalculator:
    """Calculate CSI, Doppler, DoRF, and HAR payloads for the demo window."""

    @staticmethod
    def has_payload_packets(pcap_path) -> bool:
        try:
            return pcap_path.exists() and pcap_path.stat().st_size > 24
        except Exception:
            return False

    @staticmethod
    def load_csi_capture(pcap_path, bandwidth_mhz: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        csi_data: np.ndarray | None = None
        time_pkts: np.ndarray | None = None
        nfft = int(3.2 * bandwidth_mhz)
        try:
            csi_data, time_pkts, _ = _read_ubilocate_csi(str(pcap_path), bw=bandwidth_mhz, is_4ss=True)
            if csi_data.size == 0:
                raise ValueError("No UbiLocate CSI packets found")
        except Exception:
            processor = ProcessPcap(str(pcap_path.parent), bw=bandwidth_mhz, tx_loc=[0, 0])
            csi_data_raw, time_raw, _, _ = processor.process_pcap(str(pcap_path), bw=bandwidth_mhz)
            rx_count = getattr(processor, "num_cores", 1) or 1
            nfft = processor.nfft
            csi_data = csi_data_raw.reshape((-1, nfft, rx_count, 1))
            time_pkts = np.asarray(time_raw)
        return csi_data, np.asarray(time_pkts), nfft

    @staticmethod
    def compute_ratio_payload(csi_data: np.ndarray, time_vals: np.ndarray, nfft: int) -> dict:
        packet_count = int(min(csi_data.shape[0], time_vals.size if time_vals.size else csi_data.shape[0]))
        csi_data = csi_data[:packet_count]
        time_vals = time_vals[:packet_count] if time_vals.size else np.array([])

        sampling_rate = 0.0
        if packet_count > 1 and time_vals.size == packet_count:
            duration = float(time_vals[-1] - time_vals[0])
            if duration > 0:
                sampling_rate = float((packet_count - 1) / duration)

        subcarrier_idx = min(23, nfft - 1)
        rx_count = csi_data.shape[2] if csi_data.ndim >= 3 else 1
        tx_count = csi_data.shape[3] if csi_data.ndim >= 4 else 1
        tx_pairs = list(combinations(range(tx_count), 2))

        if time_vals.size == packet_count:
            x = time_vals
            x_label = "Time (s)"
        else:
            x = np.arange(packet_count)
            x_label = "Packet index"

        series = []
        for rx_idx in range(rx_count):
            for tx_num, tx_den in tx_pairs:
                numerator = csi_data[:, subcarrier_idx, rx_idx, tx_num]
                denominator = csi_data[:, subcarrier_idx, rx_idx, tx_den]
                ratio = numerator / (denominator + 1e-12)
                series.append(
                    {
                        "rx_idx": rx_idx,
                        "tx_num": tx_num,
                        "tx_den": tx_den,
                        "x": x,
                        "x_label": x_label,
                        "ratio_mag": np.abs(ratio),
                        "ratio_phase": np.angle(ratio),
                    }
                )

        return {
            "csi_data": csi_data,
            "time_vals": time_vals,
            "packet_count": packet_count,
            "sampling_rate": sampling_rate,
            "tx_pairs": tx_pairs,
            "series": series,
            "x_label": x_label,
        }

    @staticmethod
    def root_music_csi_like(sample_data: np.ndarray, L: int = 1, num_threads: int = 1) -> np.ndarray:
        if sample_data.ndim != 2 or sample_data.shape[1] < 32:
            return np.array([], dtype=float)
        n_sc, n_t = sample_data.shape
        sig_padded = np.zeros((n_sc, n_t + 100), dtype=np.complex128)
        sig_padded[:, 50:-50] = sample_data
        sig_padded[:, :50] = sample_data[:, 50:0:-1]
        sig_padded[:, -50:] = sample_data[:, -1:-51:-1]

        windows = list(range(50, n_t + 50))

        def _estimate_window(w: int) -> np.ndarray:
            sig_window = sig_padded[:, w - 16 : w + 16]
            h = sig_window.T
            covariance = h @ h.conj().T
            covariance = np.nan_to_num(covariance)
            estimator = estimation.RootMUSIC1D(1.0)
            _, estimates = estimator.estimate(covariance, L)
            return np.asarray(estimates.locations, dtype=float)

        max_threads = max(1, os.cpu_count() or 1)
        try:
            requested_threads = int(num_threads)
        except (TypeError, ValueError):
            requested_threads = 1
        worker_count = max(1, min(requested_threads, max_threads))
        if worker_count == 1:
            doppler_vector = [_estimate_window(w) for w in windows]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                doppler_vector = list(executor.map(_estimate_window, windows))
        return np.asarray(doppler_vector, dtype=float).reshape(-1)

    @staticmethod
    def extract_csi_ratio_for_stream(csi_data: np.ndarray, rx_idx: int, tx_pair: tuple[int, int]) -> np.ndarray:
        tx_num, tx_den = tx_pair
        ratio_chunks: list[np.ndarray] = []
        max_subants = min(4, csi_data.shape[1] // 64)
        for subant in range(max_subants):
            csi_1 = csi_data[:, :, rx_idx, tx_num]
            csi_1_1 = csi_1[:, 64 * subant : 64 * (1 + subant)]
            csi_1_1 = csi_1_1[:, 6:-6]
            valid_idx = [i for i in range(csi_1_1.shape[1]) if i not in (19, 46)]
            csi_1_1 = csi_1_1[:, np.array(valid_idx, dtype=int)]

            csi_2 = csi_data[:, :, rx_idx, tx_den]
            csi_2_1 = csi_2[:, 64 * subant : 64 * (1 + subant)]
            csi_2_1 = csi_2_1[:, 6:-6]
            valid_idx = [i for i in range(csi_2_1.shape[1]) if i not in (19, 46)]
            csi_2_1 = csi_2_1[:, np.array(valid_idx, dtype=int)]

            ratio_chunks.append(csi_2_1 / (1e-6 + csi_1_1))

        if not ratio_chunks:
            return np.array([], dtype=np.complex128)
        csi_ratio = np.concatenate(ratio_chunks, axis=1)
        if csi_ratio.shape[1] < 2:
            return csi_ratio

        good_subcarriers = []
        for iii in range(csi_ratio.shape[1] - 1):
            corr = np.corrcoef(np.angle(csi_ratio[:, iii]), np.angle(csi_ratio[:, iii + 1]))[0][1]
            good_subcarriers.append(corr)
        good_subcarriers = np.abs(np.nan_to_num(np.asarray(good_subcarriers)))
        good_subcarriers = np.where(good_subcarriers > 0.6)[0]
        if good_subcarriers.size == 0:
            return np.array([], dtype=np.complex128)
        return csi_ratio[:, good_subcarriers]

    @classmethod
    def compute_doppler_payload(
        cls, csi_data: np.ndarray, time_vals: np.ndarray, packet_count: int, tx_pairs: list[tuple[int, int]], num_threads: int = 1
    ) -> dict:
        rx_count = csi_data.shape[2] if csi_data.ndim >= 3 else 1
        if time_vals.size == packet_count:
            x = np.asarray(time_vals, dtype=float)
            x_label = "Time (s)"
        else:
            x = np.arange(packet_count)
            x_label = "Packet index"

        series = []
        dopplers = []
        for rx_idx in range(rx_count):
            for tx_pair in tx_pairs:
                csi_ratio = cls.extract_csi_ratio_for_stream(csi_data, rx_idx, tx_pair)
                music_output = cls.root_music_csi_like(csi_ratio.T, num_threads=num_threads) if csi_ratio.size else np.array([])
                music_output = np.asarray(music_output, dtype=float)
                dopplers.append(music_output)
                series.append({"rx_idx": rx_idx, "tx_pair": tx_pair, "music_output": music_output, "x": x[: music_output.size], "x_label": x_label})
        return {"series": series, "dopplers": dopplers}

    @staticmethod
    def compute_dorf_payload(dopplers: list[np.ndarray], dorf_visualize: bool = False) -> dict:
        if len(dopplers) < 24:
            return {"status": "insufficient", "message": "Need 24 Doppler projections for DoRF velocity estimation."}

        selected = [np.asarray(v, dtype=float).reshape(-1) for v in dopplers[:24]]
        min_len = min((v.size for v in selected), default=0)
        if min_len == 0:
            return {"status": "insufficient", "message": "DoRF estimation skipped: at least one Doppler projection is empty."}

        doppler_matrix = np.vstack([v[:min_len] for v in selected]).T
        for i in range(doppler_matrix.shape[1]):
            doppler_matrix[:, i] = doppler_matrix[:, i] - np.mean(doppler_matrix[:, i])

        t = np.arange(doppler_matrix.shape[0])
        best_v, best_r, best_mask, best_loss, loss_hist, proj_images, _, dorf_meta = estimate_velocity_from_radial_old_dtw(
            doppler_matrix[:, :], subset_fraction=1.0, outer_iterations=10, mean_zero_velocity=False,
            true_v=None, time_axis=t, camera_numbers=list(range(doppler_matrix.shape[1])), dtw_window=8,
            use_support_dtw=False, visualise=dorf_visualize, grid_res=6, max_clusters=2, return_metadata=True,
        )

        return {
            "status": "ok", "doppler_matrix": doppler_matrix, "best_v": best_v, "best_r": best_r,
            "best_mask": best_mask, "best_loss": best_loss, "loss_hist": loss_hist,
            "proj_images": proj_images, "dorf_meta": dorf_meta,
        }

    @staticmethod
    def compute_har_payload(dorf_payload: dict) -> dict:
        if dorf_payload.get("status") != "ok":
            return {"status": "insufficient", "label": "Unknown", "scores": {"Unknown": 1.0}}

        best_v = np.asarray(dorf_payload.get("best_v"), dtype=float)
        energy = (best_v ** 2).sum(axis=1)
        mean_energy = float(np.mean(energy)) if energy.size else 0.0
        var_energy = float(np.var(energy)) if energy.size else 0.0

        scores = {
            "Stationary": max(0.0, 1.0 - min(1.0, mean_energy / 0.25)),
            "Periodic Motion": min(1.0, (var_energy + mean_energy) / 0.6),
            "Transition Motion": min(1.0, (mean_energy * 0.7 + var_energy * 0.3) / 0.9),
        }
        label = max(scores, key=scores.get)
        return {"status": "ok", "label": label, "scores": scores, "mean_energy": mean_energy, "var_energy": var_energy}
