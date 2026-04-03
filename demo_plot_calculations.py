"""Computation helpers for demo plotting pipelines."""

from __future__ import annotations

from itertools import combinations
import pickle
import sys
from pathlib import Path
from hampel import hampel
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

    _har_model = None
    _har_model_error: str | None = None

    @classmethod
    def _load_har_model(cls):
        if cls._har_model is not None:
            return cls._har_model
        if cls._har_model_error is not None:
            return None

        models_dir = Path(__file__).resolve().parent / "models"
        model_path = models_dir / "original_rocket_baseline.pkl"
        try:
            if str(models_dir) not in sys.path:
                sys.path.append(str(models_dir))
            # Ensure classifier symbols are importable for pickle payloads that
            # were serialized from scripts running as __main__.
            import __main__
            import originalrocket

            if hasattr(originalrocket, "OriginalRocketClassifier"):
                setattr(__main__, "OriginalRocketClassifier", originalrocket.OriginalRocketClassifier)
            if hasattr(originalrocket, "originalrocketclassifier"):
                setattr(__main__, "originalrocketclassifier", originalrocket.originalrocketclassifier)
            elif hasattr(originalrocket, "OriginalRocketClassifier"):
                setattr(__main__, "originalrocketclassifier", originalrocket.OriginalRocketClassifier)

            with model_path.open("rb") as f:
                cls._har_model = pickle.load(f)
            return cls._har_model
        except Exception as exc:
            cls._har_model_error = str(exc)
            return None

    @staticmethod
    def _prepare_har_input(proj_images: np.ndarray) -> np.ndarray:
        if proj_images.ndim == 2:
            proj_images = proj_images[np.newaxis, ...]
        if proj_images.ndim != 3:
            raise ValueError(f"Expected proj_images with 3 dimensions, got shape {proj_images.shape}.")

        # proj_images shape: [time, lat_bins, lon_bins] -> [samples, channels, time]
        t_len, lat_bins, lon_bins = proj_images.shape
        channels = lat_bins * lon_bins
        return proj_images.transpose(1, 2, 0).reshape(1, channels, t_len).astype(np.float32)

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
    def root_music_csi_like(sample_data: np.ndarray, L: int = 1) -> np.ndarray:
        if sample_data.ndim != 2 or sample_data.shape[1] < 32:
            return np.array([], dtype=float)
        n_sc, n_t = sample_data.shape
        sig_padded = np.zeros((n_sc, n_t + 100), dtype=np.complex128)
        sig_padded[:, 50:-50] = sample_data
        sig_padded[:, :50] = sample_data[:, 50:0:-1]
        sig_padded[:, -50:] = sample_data[:, -1:-51:-1]

        doppler_vector = []
        for w in range(50, n_t + 50):
            sig_window = sig_padded[:, w - 16 : w + 16]
            h = sig_window.T
            covariance = h @ h.conj().T
            covariance = np.nan_to_num(covariance)
            estimator = estimation.RootMUSIC1D(1.0)
            _, estimates = estimator.estimate(covariance, L)
            doppler_vector.append(estimates.locations)
        return np.asarray(doppler_vector, dtype=float).reshape(-1)

    @staticmethod
    def extract_csi_ratio_for_stream(csi_data: np.ndarray, rx_idx: int, tx_pair: tuple[int, int]) -> np.ndarray:
        tx_num, tx_den = tx_pair
        ratio_chunks: list[np.ndarray] = []
        # max_subants = min(4, csi_data.shape[1] // 64)
        # max_subants = min(4, csi_data.shape[1] // 64)
        max_subants = 1
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
    def compute_doppler_payload(cls, csi_data: np.ndarray, time_vals: np.ndarray, packet_count: int, tx_pairs: list[tuple[int, int]]) -> dict:
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
                for iii in range(np.shape(csi_ratio)[1]):
                    CSI_phase = np.angle(csi_ratio[:,iii])
                    CSI_phase = hampel(CSI_phase, 8).filtered_data
                    # CSI_phase = plot_fft_energy_and_lowpass(CSI_phase,150,5, visualize= False)['X_filt']                             
                    csi_ratio[:,iii] = 1 * np.exp(1j * CSI_phase)
                music_output = cls.root_music_csi_like(csi_ratio.T) if csi_ratio.size else np.array([])
                music_output = np.asarray(music_output, dtype=float)
                dopplers.append(music_output)
                series.append({"rx_idx": rx_idx, "tx_pair": tx_pair, "music_output": music_output, "x": x[: music_output.size], "x_label": x_label})
        return {"series": series, "dopplers": dopplers}

    @staticmethod
    def compute_dorf_payload(dopplers: list[np.ndarray], dorf_visualize: bool = False) -> dict:
        if len(dopplers) < 24:
            return {"status": "insufficient", "message": "Need 24 Doppler projections for DoRF velocity estimation."}
        def robust_window_snr(x, win=50, step=1, eps=1e-12):
                vals = []
                for i in range(0, len(x) - win + 1, step):
                    vals.append(np.std(x[i:i+win]))
                vals = np.asarray(vals)

                noise = np.percentile(vals, 10)
                signal = np.percentile(vals, 95)

                return 20 * np.log10((signal + eps) / (noise + eps))
        
    
        snr = [robust_window_snr(dopplers[j]) for j in range(len(dopplers))]
        dopplers = np.array(dopplers)[np.argsort(snr)[-12:]]
        print("Filtered Dopplers size: ", len(dopplers))
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

        proj_images = np.asarray(dorf_payload.get("proj_images"), dtype=float)
        model = DemoPlotCalculator._load_har_model()
        if model is None:
            return {
                "status": "error",
                "label": "Unknown",
                "label_value": -1,
                "scores": {"Unknown": 1.0},
                "message": f"HAR model load failed: {DemoPlotCalculator._har_model_error}",
            }

        try:
            print("proj_images shape: ", proj_images.shape)
            har_input = DemoPlotCalculator._prepare_har_input(proj_images)
            har_input = har_input / (1e-8 + np.std(har_input))
            print("har_input shape: ", har_input.shape)
            print(model.predict(har_input))
            print(model.transformer_)
            print(model.transformer_.is_fitted)
            pred_value = int(model.predict(har_input)[0])

            # Prefer explicit class-name mappings instead of calling
            # model.predict_label_names(), because some legacy pickles
            # include a typo (`calss_names_`) that can raise at inference time.
            class_names = getattr(model, "class_names_", None)
            if class_names is None:
                class_names = getattr(model, "calss_names_", None)

            if class_names is not None and pred_value < len(class_names):
                pred_label = str(class_names[pred_value])
            else:
                pred_label = str(pred_value)

            scores: dict[str, float] = {pred_label: 1.0}
            clf = getattr(model, "clf_", None)
            if clf is not None and hasattr(clf, "decision_function"):
                raw = np.asarray(clf.decision_function(model.transformer_.transform(har_input)), dtype=float).reshape(-1)
                probs = np.exp(raw - np.max(raw))
                probs = probs / (np.sum(probs) + 1e-12)
                class_names = getattr(model, "class_names_", None)
                if class_names is None:
                    class_names = getattr(model, "calss_names_", None)
                classes = np.asarray(getattr(clf, "classes_", np.arange(probs.size))).reshape(-1)
                scores = {}
                for i, p in enumerate(probs):
                    class_id = int(classes[i]) if i < classes.size else i
                    if class_names is not None and class_id < len(class_names):
                        name = str(class_names[class_id])
                    else:
                        name = str(class_id)
                    scores[name] = float(p)

            return {
                "status": "ok",
                "label": pred_label,
                "label_value": pred_value,
                "scores": scores,
            }
        except Exception as exc:
            return {
                "status": "error",
                "label": "Unknown",
                "label_value": -1,
                "scores": {"Unknown": 1.0},
                "message": f"HAR inference failed: {exc}",
            }
