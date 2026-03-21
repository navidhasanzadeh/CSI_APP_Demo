import socket
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone

NTP_PORT = 123
NTP_EPOCH_DELTA = 2208988800

DEFAULT_TIME_SERVERS = [
    "time.cloudflare.com",
    "time.google.com",
    "pool.ntp.org",
    "time.windows.com",
]


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _unix_ns_to_ntp_parts(unix_ns: int) -> tuple[int, int]:
    unix_s = unix_ns // 1_000_000_000
    ns_rem = unix_ns - unix_s * 1_000_000_000
    ntp_s = unix_s + NTP_EPOCH_DELTA
    ntp_frac = int((ns_rem * (1 << 32)) / 1_000_000_000) & 0xFFFFFFFF
    return ntp_s, ntp_frac


def _ntp_parts_to_unix_ns(ntp_s: int, ntp_frac: int) -> int:
    unix_s = ntp_s - NTP_EPOCH_DELTA
    ns = int((ntp_frac * 1_000_000_000) / (1 << 32))
    return unix_s * 1_000_000_000 + ns


def sntp_one_shot(server: str, timeout_s: float = 1.0):
    packet = bytearray(48)
    packet[0] = 0x23  # LI=0, VN=4, Mode=3 (client)
    t1_unix_ns = time.time_ns()
    t1_ntp_s, t1_ntp_frac = _unix_ns_to_ntp_parts(t1_unix_ns)
    struct.pack_into("!II", packet, 40, t1_ntp_s, t1_ntp_frac)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout_s)
        sock.sendto(packet, (server, NTP_PORT))
        data, _ = sock.recvfrom(512)
        t4_unix_ns = time.time_ns()

    if len(data) < 48:
        raise RuntimeError(f"Short NTP response from {server}: {len(data)} bytes")

    t2_ntp_s, t2_ntp_frac = struct.unpack_from("!II", data, 32)
    t3_ntp_s, t3_ntp_frac = struct.unpack_from("!II", data, 40)
    t2_unix_ns = _ntp_parts_to_unix_ns(t2_ntp_s, t2_ntp_frac)
    t3_unix_ns = _ntp_parts_to_unix_ns(t3_ntp_s, t3_ntp_frac)

    offset_ns = ((t2_unix_ns - t1_unix_ns) + (t3_unix_ns - t4_unix_ns)) // 2
    delay_ns = (t4_unix_ns - t1_unix_ns) - (t3_unix_ns - t2_unix_ns)
    return offset_ns, delay_ns, t4_unix_ns


def best_startup_sync(servers, attempts_per_server=3, timeout_s=1.0):
    best = None
    last_err = None
    for server in servers:
        if not server:
            continue
        for _ in range(attempts_per_server):
            try:
                offset_ns, delay_ns, t4_unix_ns = sntp_one_shot(server, timeout_s=timeout_s)
                sample = {
                    "server": server,
                    "offset_ns": offset_ns,
                    "delay_ns": delay_ns,
                    "t4_unix_ns": t4_unix_ns,
                    "mono_at_t4_ns": time.monotonic_ns(),
                }
                if best is None or sample["delay_ns"] < best["delay_ns"]:
                    best = sample
            except Exception as exc:
                last_err = exc
    if best is None:
        raise RuntimeError(f"All SNTP attempts failed. Last error: {last_err}")
    return best


def fmt_utc_from_ns(unix_ns: int) -> str:
    ms = (unix_ns // 1_000_000) % 1000
    dt = datetime.fromtimestamp(unix_ns / 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{ms:03d} UTC"


def fmt_local_from_ns(unix_ns: int) -> str:
    ms = (unix_ns // 1_000_000) % 1000
    dt = datetime.fromtimestamp(unix_ns / 1_000_000_000)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{ms:03d}"


@dataclass
class TimeReference:
    enabled: bool = False
    server: str | None = None
    fallback_servers: list[str] | None = None
    offset_ns: int | None = None
    delay_ns: int | None = None
    ref_utc_ns: int | None = None
    mono_ref_ns: int | None = None
    sync_server: str | None = None
    last_error: str | None = None

    def sync(self, *, attempts_per_server: int = 3, timeout_s: float = 1.0) -> bool:
        if not self.enabled:
            return False
        servers = [self.server] if self.server else None
        if not servers:
            servers = self.fallback_servers or DEFAULT_TIME_SERVERS
        try:
            best = best_startup_sync(servers, attempts_per_server=attempts_per_server, timeout_s=timeout_s)
        except Exception as exc:
            self.last_error = str(exc)
            return False

        self.offset_ns = best["offset_ns"]
        self.delay_ns = best["delay_ns"]
        self.sync_server = best["server"]
        self.ref_utc_ns = best["t4_unix_ns"] + best["offset_ns"]
        self.mono_ref_ns = best["mono_at_t4_ns"]
        self.last_error = None
        return True

    def now_ns(self) -> int:
        if self.enabled and self.ref_utc_ns is not None and self.mono_ref_ns is not None:
            return self.ref_utc_ns + (time.monotonic_ns() - self.mono_ref_ns)
        return time.time_ns()

    def now(self) -> float:
        return self.now_ns() / 1_000_000_000

    def now_datetime(self, tz=None) -> datetime:
        return datetime.fromtimestamp(self.now(), tz=tz)

    def offset_ms(self) -> float | None:
        if self.offset_ns is None:
            return None
        return self.offset_ns / 1e6


_GLOBAL_REFERENCE = TimeReference()


def build_time_reference(profile: dict | None) -> TimeReference:
    profile = profile or {}
    enabled = _as_bool(profile.get("use_time_server", False))
    server = (profile.get("time_server") or "").strip() or None
    ref = TimeReference(enabled=enabled, server=server, fallback_servers=DEFAULT_TIME_SERVERS)
    if enabled:
        ref.sync()
    return ref


def set_global_time_reference(ref: TimeReference) -> None:
    global _GLOBAL_REFERENCE
    _GLOBAL_REFERENCE = ref


def get_global_time_reference() -> TimeReference:
    return _GLOBAL_REFERENCE


def now_ns() -> int:
    return _GLOBAL_REFERENCE.now_ns()


def now() -> float:
    return _GLOBAL_REFERENCE.now()


def now_datetime(tz=None) -> datetime:
    return _GLOBAL_REFERENCE.now_datetime(tz=tz)
