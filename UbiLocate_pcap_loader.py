import numpy as np
import struct
import matplotlib.pyplot as plt
import os

# ==========================================
# Part 1: C-MEX Port (unpack_float)
# ==========================================
def unpack_float(H_uint32, nfft):
    """
    Python port of unpack_float.c (Nexmon CSI extractor).
    Decodes the compressed CSI format into complex numbers.
    """
    nbits = 10
    nman = 12
    nexp = 6
    
    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = (1 << (nexp - 1))
    
    vi = (H_uint32 >> (nexp + nman)) & iq_mask
    vq = (H_uint32 >> nexp) & iq_mask
    e = (H_uint32 & e_mask).astype(np.int8) 
    
    e[e >= e_p] -= (e_p * 2)
    
    x = vi | vq
    
    bit_len = np.zeros_like(x, dtype=np.int32)
    mask_nz = x > 0
    float_x = x.astype(np.float64)
    float_x[float_x == 0] = 1 
    bits = np.floor(np.log2(float_x)).astype(np.int32) + 1
    bits[~mask_nz] = 0
    
    temp_e = e.astype(np.int32) + bits
    maxbit = np.max(temp_e)
    shft = nbits - maxbit
    final_e = e + shft
    
    sgnr_mask = (1 << (nexp + 2 * nman - 1))
    sgni_mask = (sgnr_mask >> nman)
    
    sign_i = np.ones_like(vi, dtype=np.int32)
    sign_i[(H_uint32 & sgnr_mask) != 0] = -1
    
    sign_q = np.ones_like(vq, dtype=np.int32)
    sign_q[(H_uint32 & sgni_mask) != 0] = -1
    
    def apply_shift(val, exp):
        res = np.zeros_like(val)
        mask_neg = (exp < 0) & (exp >= -nman)
        res[mask_neg] = val[mask_neg] >> -exp[mask_neg]
        mask_pos = exp >= 0
        res[mask_pos] = val[mask_pos] << exp[mask_pos]
        return res

    final_vi = apply_shift(vi, final_e) * sign_i
    final_vq = apply_shift(vq, final_e) * sign_q
    
    return final_vi + 1j * final_vq

# ==========================================
# Part 2: PCAP Reader
# ==========================================
class PcapReader:
    def __init__(self, filename):
        self.filename = filename
        self.f = None
        self.file_size = 0
        
    def open(self):
        self.f = open(self.filename, 'rb')
        self.f.seek(0, 2)
        self.file_size = self.f.tell()
        self.f.seek(0)
        
        # Global Header
        header = self.f.read(24)
        if len(header) < 24:
            raise ValueError("File too short for PCAP header")

    def next_frame(self):
        # Frame Header
        header_bytes = self.f.read(16)
        if not header_bytes or len(header_bytes) < 16:
            return None
        
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack('<IIII', header_bytes)
        payload = self.f.read(incl_len)
        if len(payload) < incl_len:
            return None
            
        return {
            'ts_sec': ts_sec,
            'ts_usec': ts_usec,
            'orig_len': orig_len,
            'payload': payload
        }

    def close(self):
        if self.f:
            self.f.close()

# ==========================================
# Part 3: Main Processing Logic
# ==========================================
def read_csi_data(filename, bw=80, is_4ss=False):
    """
    Reads CSI data and timestamps from PCAP.
    Returns: (csi_matrix, timestamps)
    """
    HOFFSET = 16 
    NFFT = int(bw * 3.2) 
    
    reader = PcapReader(filename)
    reader.open()
    
    prevfwcnt = -1
    
    # Configuration
    if is_4ss:
        nss_config = 4
        rxcore_config = 4
        mask_toprocess = nss_config * rxcore_config
        processedmask = 0
        slice_buffer = None
    else:
        mask_toprocess = 4 
        processedcore = 0
        slice_buffer = None
        rxcore_config = 1 

    csi_list = []
    timestamps_list = []
    
    # Temporary variable to hold the timestamp of the current packet being assembled
    current_packet_ts = 0.0
    
    while True:
        frame = reader.next_frame()
        if frame is None:
            break
            
        payload = frame['payload']
        # Compute exact timestamp for this frame
        frame_ts = float(frame['ts_sec']) + float(frame['ts_usec']) * 1e-6

        if len(payload) < (HOFFSET + NFFT) * 4:
            continue
            
        payload_u32 = np.frombuffer(payload, dtype=np.uint32)
        if len(payload_u32) < 16: continue

        val_15 = payload_u32[14]
        val_14 = payload_u32[13]
        
        fwcnt = (val_15 >> 16) & 0xFFFF
        
        if is_4ss:
            fwmask = (val_14 >> 16) & 0xFF
            current_idx = fwmask
            
            if fwcnt > prevfwcnt:
                processedmask = 0
                prevfwcnt = fwcnt
                # Start new packet buffer
                slice_buffer = np.zeros((len(payload_u32), 16), dtype=np.uint32)
                # Capture timestamp of the first frame encountered for this new packet ID
                current_packet_ts = frame_ts
            
            processedmask += 1
            
            if slice_buffer is not None and current_idx < 16:
                 slice_buffer[:, current_idx] = payload_u32
                 
            if processedmask == mask_toprocess:
                csi_matrix = np.zeros((NFFT, mask_toprocess), dtype=np.complex128)
                valid_extraction = True
                
                for jj in range(mask_toprocess):
                    col_data = slice_buffer[:, jj]
                    H = col_data[15 : 15+NFFT] 
                    if len(H) == NFFT:
                        c_num = unpack_float(H, NFFT)
                        c_num = np.fft.fftshift(c_num)
                        csi_matrix[:, jj] = c_num
                    else:
                        valid_extraction = False
                
                if valid_extraction:
                    csi_list.append(csi_matrix)
                    timestamps_list.append(current_packet_ts)
                
        else:
            # 1SS Logic
            tmp = (val_14 >> 16) & 0xFF
            rxcore = (tmp // 4) 
            
            if fwcnt > prevfwcnt:
                processedcore = 0
                prevfwcnt = fwcnt
                slice_buffer = np.zeros((len(payload_u32), 4), dtype=np.uint32)
                current_packet_ts = frame_ts
            
            processedcore += 1
            
            if slice_buffer is not None and rxcore < 4:
                slice_buffer[:, rxcore] = payload_u32
                
            if processedcore == 4:
                csi_matrix = np.zeros((NFFT, 4), dtype=np.complex128)
                valid_extraction = True
                for jj in range(4):
                    col_data = slice_buffer[:, jj]
                    H = col_data[15 : 15+NFFT]
                    if len(H) == NFFT:
                        c_num = unpack_float(H, NFFT)
                        c_num = np.fft.fftshift(c_num)
                        csi_matrix[:, jj] = c_num
                    else:
                        valid_extraction = False
                
                if valid_extraction:
                    csi_list.append(csi_matrix)
                    timestamps_list.append(current_packet_ts)

    reader.close()
    
    if not csi_list:
        print("No valid CSI packets found.")
        return None, None

    return np.array(csi_list), np.array(timestamps_list)

# ==========================================
# Part 4: Plotting Drivers
# ==========================================

def run_example_1ss(file_path):
    print(f"Processing 1SS file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None

    # Load Data
    csi_data, timestamps = read_csi_data(file_path, bw=80, is_4ss=False)
    
    if csi_data is None: return None, None

    print(f"Loaded {csi_data.shape[0]} packets.")
    
    # Normalize timestamps to start at 0
    time_rel = timestamps - timestamps[0]

    # --- Plot 1: Packet Snapshot (Original) ---
    plt.figure(figsize=(10, 8))
    plt.suptitle("Snapshot: CSI Amplitude (1SS) - Packet 1")
    packet_idx = 0
    for ii in range(4):
        plt.subplot(4, 1, ii + 1)
        plt.plot(np.abs(csi_data[packet_idx, :, ii]))
        plt.ylabel(f'Core {ii+1}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- Plot 2: Amplitude vs Time (New) ---
    # We pick a central subcarrier (e.g., 128) to visualize over time
    subcarrier_idx = 27
    
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"CSI Amplitude vs Time (Subcarrier {subcarrier_idx})")
    
    for ii in range(4):
        plt.subplot(4, 1, ii + 1)
        # Extract amplitude for specific core across all packets
        # csi_data shape: [Packets, NFFT, Cores]
        amp_over_time = np.abs(csi_data[:, subcarrier_idx, ii])
        
        plt.plot(time_rel, amp_over_time)
        plt.ylabel(f'Core {ii+1} Amp')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()
    
    return csi_data, timestamps


def run_example_4ss(file_path):
    print(f"Processing 4SS file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None, None

    # Load Data
    cmplxall_raw_all, timestamps = read_csi_data(file_path, bw=80, is_4ss=True)
    
    if cmplxall_raw_all is None: return None, None

    print(f"Loaded {cmplxall_raw_all.shape[0]} packets.")
    
    packets, nfft, flattened_streams = cmplxall_raw_all.shape
    N = 4 # Rx
    SS = 4 # Tx
    
    # Reshape: (Packets, 256, Rx, Tx)
    csi_reshaped = np.zeros((packets, nfft, N, SS), dtype=np.complex128)
    for ii in range(N):
        start = ii * SS
        end = (ii + 1) * SS
        csi_reshaped[:, :, ii, :] = cmplxall_raw_all[:, :, start:end]

    # Normalize timestamps
    time_rel = timestamps - timestamps[0]

    # --- Plot 1: Packet Snapshot (Original) ---
    plt.figure(figsize=(12, 10))
    plt.suptitle("Snapshot: CSI Amplitude (4SS) - Packet 1")
    counter = 1
    packet_idx = 0
    for rx in range(4):
        for tx in range(4):
            plt.subplot(4, 4, counter)
            plt.plot(np.abs(csi_reshaped[packet_idx, :, rx, tx]))
            plt.title(f"Tx: {tx+1}, Rx: {rx+1}", fontsize=8)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            counter += 1
    plt.tight_layout()
    plt.show()
    
    # --- Plot 2: Amplitude vs Time (New) ---
    subcarrier_idx = 27
    
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"CSI Amplitude vs Time (Subcarrier {subcarrier_idx})")
    counter = 1
    
    for rx in range(4):
        for tx in range(4):
            plt.subplot(4, 4, counter)
            # Slice: All packets, specific subcarrier, specific Rx, specific Tx
            amp_over_time = np.abs(csi_reshaped[:, subcarrier_idx, rx, tx])
            
            plt.plot(time_rel, amp_over_time)
            plt.title(f"Tx: {tx+1}, Rx: {rx+1}", fontsize=8)
            
            # Only label axes on the edges to reduce clutter
            if rx == 3: plt.xlabel('Time (s)')
            else: plt.tick_params(labelbottom=False)
            
            if tx == 0: plt.ylabel('Amp')
            else: plt.tick_params(labelleft=False)
            
            plt.grid(True)
            counter += 1
            
    plt.tight_layout()
    plt.show()
    
    return csi_reshaped, timestamps

# ==========================================
# Main Entry Point
# ==========================================

# Example Usage
file_4ss = "csi13.pcap" 

# Now returns data and timestamps
csi_data, timestamps = run_example_4ss(file_4ss)

if timestamps is not None:
    print(f"Time duration: {timestamps[-1] - timestamps[0]:.2f} seconds")