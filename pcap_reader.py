import glob
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from tqdm import tqdm
from csiread import Nexmon

try:  # pragma: no cover - optional dependency at runtime
    from hampel import hampel
except ImportError:  # pragma: no cover - optional dependency at runtime
    hampel = None

try:  # pragma: no cover - optional dependency at runtime
    from sklearn.linear_model import RANSACRegressor
except ImportError:  # pragma: no cover - optional dependency at runtime
    RANSACRegressor = None

# Optional: Force interactive backend (uncomment if needed)
# import matplotlib
# matplotlib.use('TkAgg')  # Or 'Qt5Agg'

class SignalAligner:
    """
    An interactive tool to visualize multiple signals and manually align segmentation
    points (start/end) by adjusting a common delay.
    """
    def __init__(self, signals, time_points_list, start_times, end_times):
        """
        Initializes the aligner and sets up the plot.

        Args:
            signals (list[np.ndarray]): A list of signals to plot.
            time_points_list (list[np.ndarray]): A list of corresponding time points for each signal.
            start_times (np.ndarray): Array of time values where segments start.
            end_times (np.ndarray): Array of time values where segments end.
        """
        # --- Store initial data ---
        self.signals = signals
        self.time_points_list = time_points_list
        self.original_start_times = np.array(start_times)
        self.original_end_times = np.array(end_times)

        # --- State variables ---
        self.current_delay = 0.0
        self.output = None

        # --- Setup the figure and subplots for each signal ---
        num_signals = len(signals)
        self.fig, self.axs = plt.subplots(num_signals, 1, sharex=True, figsize=(12, 3 * num_signals))
        # Ensure self.axs is always an array for consistent handling
        if num_signals == 1:
            self.axs = [self.axs]
        plt.subplots_adjust(bottom=0.25)

        self.fig.suptitle('Interactive Signal Aligner', fontsize=16)

        # --- Draw signals and initial vertical lines ---
        self.start_lines = [[] for _ in range(num_signals)]
        self.end_lines = [[] for _ in range(num_signals)]

        for i, ax in enumerate(self.axs):
            ax.plot(self.time_points_list[i], self.signals[i], label=f'Signal {i+1}')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            ax.legend(loc='center')

            for t in self.original_start_times:
                line = ax.axvline(t, color='g', linestyle='--', label='Start Points' if not self.start_lines[i] else "")
                self.start_lines[i].append(line)

            for t in self.original_end_times:
                line = ax.axvline(t, color='r', linestyle='--', label='End Points' if not self.end_lines[i] else "")
                self.end_lines[i].append(line)
        
        self.axs[-1].set_xlabel('Time')


        # --- Create and configure widgets ---
        slider_ax = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        # Determine max delay from the signal with the widest time range
        max_time = max(tp[-1] for tp in self.time_points_list)
        min_time = min(tp[0] for tp in self.time_points_list)
        max_delay = (max_time - min_time) / 4
        
        self.delay_slider = Slider(
            ax=slider_ax,
            label='Delay',
            valmin=-max_delay,
            valmax=max_delay,
            valinit=0.0,
        )

        self.delay_text = self.fig.text(0.5, 0.15, f'Current Delay: {self.current_delay:.4f}', ha='center')
        save_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.save_button = Button(save_ax, 'Save', hovercolor='0.975')

        # --- Connect widgets to callback functions ---
        self.delay_slider.on_changed(self._update_delay)
        self.save_button.on_clicked(self._save_and_close)
        # Add a handler for the window close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        """Callback for when the figure window is closed."""
        if self.output is None:
            print("Window closed without saving. Returning None.")

    def _update_delay(self, delay_value):
        """Callback function to update line positions when slider moves."""
        self.current_delay = delay_value
        
        # Update start and end lines on all subplots
        for i in range(len(self.signals)):
            for j, t in enumerate(self.original_start_times):
                self.start_lines[i][j].set_xdata([t + self.current_delay])
            for j, t in enumerate(self.original_end_times):
                self.end_lines[i][j].set_xdata([t + self.current_delay])
            
        self.delay_text.set_text(f'Current Delay: {self.current_delay:.4f}')
        self.fig.canvas.draw_idle()

    def _save_and_close(self, event):
        """Callback to save results, extract windows, and close the plot."""
        new_start_times = self.original_start_times + self.current_delay
        new_end_times = self.original_end_times + self.current_delay

        segmented_windows = []
        
        for i, signal in enumerate(self.signals):
            time_pts = self.time_points_list[i]
            signal_windows = []            
            start_indices = np.searchsorted(time_pts, new_start_times, side='left')
            end_indices = np.searchsorted(time_pts, new_end_times, side='left')
            
            start_indices = np.clip(start_indices, 0, len(signal) - 1)
            end_indices = np.clip(end_indices, 0, len(signal) - 1)
            
            for start_idx, end_idx in zip(start_indices, end_indices):
                if end_idx > start_idx:
                    window = signal[start_idx:end_idx]
                    signal_windows.append((window, (start_idx,end_idx)))
                else:
                    signal_windows.append(np.array([], (None,None))) # Invalid window
            segmented_windows.append(signal_windows)

        self.output = {
            'delay': self.current_delay,
            'new_start_times': new_start_times,
            'new_end_times': new_end_times,
            'segmented_windows': segmented_windows
        }
        
        print("Data saved. Closing window.")
        plt.close(self.fig)

    def align(self):
        """
        Displays the plot and waits for the user to finish.
        
        Returns:
            dict: A dictionary containing the delay, new start/end times, and segmented windows.
                  Returns None if the window is closed without saving.
        """
        plt.show(block=True)  # Explicitly block to ensure waiting
        return self.output

def align_signal_points(signals, time_points_list, start_times, end_times):
    """
    A wrapper function to launch the SignalAligner UI for multiple signals.

    Args:
        signals (list[np.ndarray]): A list of signals to plot.
        time_points_list (list[np.ndarray]): A list of corresponding time points for each signal.
        start_times (np.ndarray): Array of time values where segments start.
        end_times (np.ndarray): Array of time values where segments end.

    Returns:
        dict: A dictionary with the results after the user clicks 'Save'.
    """
    aligner = SignalAligner(signals, time_points_list, start_times, end_times)
    return aligner.align()


def moving_average(x, w):
    l = len(x)
    x = np.concatenate([x[::-1], x, x[::-1]])
    return np.convolve(x, np.ones(w), 'same')[l:-l] / w


def P2R(r, theta):
    return r * np.exp(1j * theta)


def ant_processing(x: np.ndarray) -> np.ndarray:
    if RANSACRegressor is None:
        raise ImportError("scikit-learn is required for antenna phase processing")
    if hampel is None:
        raise ImportError("hampel is required for antenna phase processing")

    x = x.copy()
    for i in range(len(x.T)):
        mag = np.abs(x[:, i])
        phase = np.unwrap(np.angle(x[:, i]))
        phase_unwrap = phase
        x_k = np.arange(len(phase)).reshape(-1,)

        ransac = RANSACRegressor(random_state=42)
        ransac.fit(x_k.reshape(-1, 1), phase_unwrap.reshape(-1,))
        y_k = ransac.predict(x_k.reshape(-1, 1))
        phase_unwrap = phase_unwrap - y_k
        phase_new = phase_unwrap
        x[:, i] = P2R(mag, phase_new)
    for i in range(len(x)):
        phase = np.angle(x[i, :])
        mag = np.abs(x[i, :])
        phase_unwrap = hampel(phase, window_size=4).filtered_data
        x[i, :] = P2R(mag, phase_unwrap)
    return x

def seq2index(csi_seq, seq_target):
    target_1 = []
    target_2 = []
    # csi_seq = [csi_data.frames[i].sequence_no for i in range(len(csi_data.frames))]
    idx_ptr = 0
    
    for i in tqdm(range(len(seq_target))):
        idx_ptr = idx_ptr + csi_seq[idx_ptr:].index(seq_target[i])
        target_1.append(idx_ptr)
    return target_1

class ProcessPcap:
    def __init__(self, exp_dir, bw, tx_loc):
        self.core_to_antenna = {
            1: 0,
            3: 1,
            0: 2
        }
        self.exp_dir = exp_dir
        self.bw = bw
        self.nfft = int(3.2 * bw)
        self.num_cores = len(self.core_to_antenna.keys())
        self.tx_loc = tx_loc
    def get_data(self):
        # pcap_files = glob.glob(f"{self.exp_dir}/*.pcap")
        pcap_files = glob.glob("{}/*.pcap".format(self.exp_dir))
        data_dict = {}
        for pcap_file in pcap_files:
            # ap_no = int(os.path.basename(pcap_file).split('.')[0]) # Get AP number from filename
            ap_no = 0
            csi_data, time_pkts, seq, mac_addrs = self.process_pcap(pcap_file, bw=80)
            # Get time info
            # time = float(open("{}/{}_time.txt".format(self.exp_dir, ap_no), "r").readlines()[0])
            # time *= 1e-3
            # loc_and_ori = open("{}/{}_survey.txt".format(self.exp_dir, ap_no), "r").readlines()
            data_dict["ap_{}".format(ap_no)] = csi_data
            data_dict["ap_{}_time".format(ap_no)] = time_pkts
            # data_dict["ap_{}_loc".format(ap_no)] = [float(loc_and_ori[0]), float(loc_and_ori[1]), float(loc_and_ori[2])]
            # data_dict["ap_{}_ori".format(ap_no)] = np.deg2rad(float(loc_and_ori[3]))
            data_dict["ap_{}_seq".format(ap_no)] = seq
            data_dict["ap_{}_mac".format(ap_no)] = mac_addrs
       
        # meta_data = open("{}/param.txt".format(self.exp_dir), "r").readlines()
        data_dict["CH_NO"] = 0
        data_dict["BW"] = 0
        data_dict["MAC_ADDR"] = np.unique(mac_addrs) if len(mac_addrs) > 0 else 0
        data_dict["TX_LOC"] = self.tx_loc
        return data_dict
    def process_pcap(self, pcap_file, bw=20):
        csidata_fast = Nexmon(pcap_file, chip='4366c0', bw=bw, if_report=False)
        csidata_fast.read()
        csi_data = csidata_fast.csi
        antennas, time_pkts, seq, mac_addrs = self.get_meta_info(pcap_file, csi_data.shape[0])
        # Remove packets collected from internal core
        packets_to_delete = np.where(antennas == -1)
        csi_data = np.delete(csi_data, packets_to_delete, axis=0)
        antennas = np.delete(antennas, packets_to_delete)
        # Apply FFT Shift
        # csi_data = np.fft.fftshift(csi_data, axes=1) # Need to figure out the axis
        # Club packets based on antennas to form 192 dim vector per packet
        total_pkts = csi_data.shape[0]
        csi_clubbed = np.zeros((total_pkts, self.num_cores * self.nfft), dtype=np.complex_)
        pkt_idx = 0
        cores_written = 0
        for i in range(total_pkts):
            ant = int(antennas[i])
            csi_clubbed[pkt_idx, ant * self.nfft:(ant + 1) * self.nfft] = csi_data[i]
            cores_written += 1
            if cores_written == self.num_cores:
                cores_written = 0
                pkt_idx += 1
        csi_data = csi_clubbed[:pkt_idx] # Remove empty indices
        time_pkts = time_pkts - time_pkts[0] # Start timer at 0
        return csi_data, time_pkts, seq, mac_addrs
    def get_meta_info(self, pcap_file, num_pkts):
        count = 0
        cores_written = 0
        time_prev = 0 # to account for wrapping in [0, 10e5]
        time_jump = 0
        packet_idx = 0
        ant = np.zeros([num_pkts]) - 1
        time_pkt = np.zeros([num_pkts])
        sequence_number = np.zeros([num_pkts])
        mac_addrs = []
        f = open(pcap_file, 'rb')
        endian = self.__pcapheader(f)
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            sec, usec, caplen, wirelen = struct.unpack(endian + "IIII", hdr)
            buf = f.read(42)
            if buf[6:12] != b'NEXMON':
                f.seek(caplen - 42, os.SEEK_CUR)
                continue
            if count == 0:
                time_curr_packet = usec
            if cores_written == 0:
                time_curr_packet = float(usec) + time_jump * 1e6
                if time_curr_packet < time_prev:
                    time_curr_packet = time_curr_packet + 1e6
                    time_jump += 1
                time_prev = time_curr_packet
            frame = f.read(18)
            if len(frame) < 18:
                print("PROBLEM")
                _ = f.read(caplen - 42 - 18)
                continue
            magic, src_addr, seq, core_spatial, chan_spec, chip_version = struct.unpack(
                endian + "I6sHHHH", frame
            )
            core = (core_spatial >> 8) & 0x3
            _ = f.read(caplen - 42 - 18)
            # Skip data from core 2 (internal core)
            if core not in self.core_to_antenna:
                count += 1
                continue
            ant[count] = self.core_to_antenna[core]
            cores_written += 1
            if cores_written == len(self.core_to_antenna.keys()):
                mac_str = ':'.join(f"{b:02x}" for b in src_addr)
                mac_addrs.append(mac_str)
                cores_written = 0
                time_pkt[packet_idx] = time_curr_packet * 1e-6
                sequence_number[packet_idx] = seq
                packet_idx += 1
            count += 1
        time_pkt = time_pkt[:packet_idx]
        sequence_number = sequence_number[:packet_idx]
        mac_addrs = np.array(mac_addrs)
        return ant, time_pkt, sequence_number, mac_addrs
    def __pcapheader(self, f):
        magic = f.read(4)
        if magic == b"\xa1\xb2\xc3\xd4": # big endian
            endian = ">"
            self.nano = False
        elif magic == b"\xd4\xc3\xb2\xa1": # little endian
            endian = "<"
            self.nano = False
        elif magic == b"\xa1\xb2\x3c\x4d": # big endian, nanosecond-precision
            endian = ">"
            self.nano = True
        elif magic == b"\x4d\x3c\xb2\xa1": # little endian, nanosecond-precision
            endian = "<"
            self.nano = True
        else:
            raise Exception("Not a pcap capture file (bad magic: %r)" % magic)
        f.seek(20, os.SEEK_CUR)
        return endian


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.modules.setdefault("pcap_reader", sys.modules[__name__])
    src_path = Path(__file__).resolve().parent / "src"
    app_path = src_path / "app"
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(app_path))
    from pcap_reader_ui import launch_viewer

    launch_viewer()
