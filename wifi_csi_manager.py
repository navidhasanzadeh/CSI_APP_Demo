"""Shared utilities for Wi-Fi CSI collection and router interactions."""
from __future__ import annotations

import datetime
import ftplib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple

import paramiko
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox
from scp import SCPClient

from packet_counter import count_packets_for_macs
import time_reference as time_reference_module

DEFAULT_REMOTE_DIR = "/mnt/CSI_USB/"
SCRIPT_PROFILE = "nexmon"
SCRIPT_CATEGORY = "sniffer"
SCRIPT_BASE_DIR = Path(__file__).parent / "scripts"


class WiFiRouter:
    """Router helper for CSI collection tasks."""

    def __init__(
        self,
        router_ip,
        router_username,
        router_password,
        key_filename=None,
        debug=False,
        ftp_host=None,
        ftp_user=None,
        ftp_password=None,
        ssh_timeout: float | None = None,
        *,
        script_profile: str | None = None,
        script_category: str | None = None,
    ):
        self.router_ip = router_ip
        self.router_username = router_username
        self.router_password = router_password
        self.key_filename = key_filename
        self.debug = debug
        self.ssh_timeout = ssh_timeout

        self.script_profile = (script_profile or SCRIPT_PROFILE).strip() or SCRIPT_PROFILE
        self.script_category = (script_category or SCRIPT_CATEGORY).strip() or SCRIPT_CATEGORY

        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_password = ftp_password

        self.ssh = None
        self.sending = False
        self.interface = None
        self.capture_thread = None

        self._init_ssh_connection()

    def _init_ssh_connection(self):
        if self.debug:
            print(f"Initializing SSH connection to {self.router_ip}...")

        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh.connect(
            self.router_ip,
            username=self.router_username,
            password=self.router_password,
            key_filename=self.key_filename,
            timeout=self.ssh_timeout,
            banner_timeout=self.ssh_timeout,
            auth_timeout=self.ssh_timeout,
        )

        if self.debug:
            print("SSH connection established.")

    def send_command(self, cmd, sleep=0, get_pty=False):
        if self.debug:
            print(f"Executing command: {cmd}")

        _, ssh_stdout, _ = self.ssh.exec_command(cmd, get_pty=get_pty)
        output = ssh_stdout.readlines()
        time.sleep(sleep)
        return output

    def _load_script_commands(self, script_name: str):
        script_path = (
            SCRIPT_BASE_DIR
            / self.script_profile
            / self.script_category
            / f"{script_name}.json"
        )
        if not script_path.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")

        with open(script_path, "r", encoding="utf-8") as script_file:
            return json.load(script_file)

    def _execute_script_commands(self, script_name: str, context: dict) -> dict:
        commands = self._load_script_commands(script_name)
        context_data = dict(context)

        def _render_value(value):
            if isinstance(value, str):
                return value.format(**context_data)
            return value

        for entry in sorted(commands, key=lambda item: item.get("order", 0)):
            when_key = entry.get("when")
            if when_key and not context_data.get(when_key):
                continue

            sleep_value_raw = entry.get("sleep", 0)
            sleep_value = float(_render_value(str(sleep_value_raw))) if sleep_value_raw else 0
            command_template = entry.get("command")

            if command_template:
                command = command_template.format(**context_data)
                if self.debug:
                    print(
                        f"[{entry.get('label')}] {entry.get('description')}\n"
                        f"Command: {command}\nSleep: {sleep_value}"
                    )
                output = self.send_command(command, sleep=sleep_value)
                store_key = entry.get("store_output_as")
                if store_key is not None:
                    context_data[store_key] = output[0].strip() if output else ""
            else:
                if self.debug:
                    print(
                        f"[{entry.get('label')}] {entry.get('description')}\n"
                        f"Sleep: {sleep_value} (no command executed)"
                    )
                time.sleep(sleep_value)

        return context_data

    def transfer_file(self, remote_path, local_path, use_ftp=False):
        if use_ftp:
            if self.debug:
                print(f"Transferring via FTP from {remote_path} to {local_path}...")
            if not all([self.ftp_host, self.ftp_user, self.ftp_password]):
                raise RuntimeError("FTP credentials or host missing. Cannot transfer via FTP.")

            try:
                with ftplib.FTP(self.ftp_host) as ftp:
                    ftp.login(self.ftp_user, self.ftp_password)
                    ftp.set_pasv(False)
                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {remote_path}", f.write)
            except ftplib.all_errors as exc:
                raise RuntimeError(f"FTP transfer failed: {exc}") from exc

            if self.debug:
                print(f"File transfer (FTP) complete: {remote_path} -> {local_path}")
        else:
            if self.debug:
                print(f"Transferring via SCP from {remote_path} to {local_path}...")

            with SCPClient(self.ssh.get_transport()) as scp:
                scp.get(f"/mnt/{remote_path}", local_path)

            if self.debug:
                print(f"File transfer (SCP) complete: {remote_path} -> {local_path}")

    def setup_router(self):
        now_ns = time_reference_module.get_global_time_reference().now_ns()
        seconds = now_ns // 1_000_000_000
        nanoseconds = now_ns % 1_000_000_000
        print("Wi-Fi router time: ", seconds, nanoseconds)
        self._execute_script_commands(
            "setup_router", {"seconds": seconds, "nanoseconds": nanoseconds}
        )

    def set_power(self, value=100, reboot=False):
        self._execute_script_commands(
            "set_power",
            {
                "value": value,
                "reboot": reboot,
            },
        )

    def configure_monitor_mode(self, channel="6/20", core=15, mac_addresses=None):
        freq_str = channel.split("/")[0]
        try:
            freq = int(freq_str)
        except ValueError:
            freq = 6

        self.interface = "eth5" if freq < 30 else "eth6"

        if self.debug:
            print(f"Channel: {channel}, freq={freq} → Using interface: {self.interface}")

        mac_args = ""
        if mac_addresses:
            if isinstance(mac_addresses, str):
                mac_args = f" -m {mac_addresses}"
            elif isinstance(mac_addresses, list):
                for mac in mac_addresses:
                    mac_args += f" -m {mac}"
        result_context = self._execute_script_commands(
            "configure_monitor_mode",
            {
                "interface": self.interface,
                "channel": channel,
                "core": core,
                "mac_args": mac_args,
            },
        )

        return result_context.get("mcp_code", "")

    def _capture_csi(self, duration, remote_directory, exp_name, delete_prev_pcap):
        self.sending = True

        try:
            transport = self.ssh.get_transport() if self.ssh else None
            if not transport or not transport.is_active():
                raise RuntimeError("SSH connection is not active.")

            if not self.interface:
                raise RuntimeError("Interface not configured. Call configure_monitor_mode() first.")

            date_str = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{exp_name}_{date_str}.pcap"
            full_remote_path = f"{remote_directory}/{filename}"

            if self.debug:
                print(
                    f"Starting CSI capture on {self.interface}...\n"
                    f"Remote pcap file: {full_remote_path}\n"
                    f"Capture duration: {duration} seconds\n"
                    f"delete_prev_pcap={delete_prev_pcap}"
                )

            # cmd_nexutil_phase = f"cd /jffs; ./nexutil -I{self.interface} -s526 -i -l4 -v1"
            # self.send_command(cmd_nexutil_phase, sleep=0)
            self._execute_script_commands(
                "capture_csi",
                {
                    "interface": self.interface,
                    "remote_directory": remote_directory,
                    "full_remote_path": full_remote_path,
                    "delete_prev_pcap": delete_prev_pcap,
                    "capture_duration": duration,
                },
            )

            if self.debug:
                print("tcpdump killed automatically after duration.")

            return full_remote_path
        except (EOFError, paramiko.SSHException, RuntimeError) as exc:
            if self.debug:
                print(f"CSI capture aborted: {exc}")
            return None
        finally:
            self.sending = False

    def start_csi_capture_in_thread(
        self,
        duration=10,
        remote_directory="/mnt/CSI_USB",
        exp_name="Test_CSI_1_",
        delete_prev_pcap=False,
    ):
        if self.debug:
            print("Spawning CSI capture thread...")

        if self.capture_thread and self.capture_thread.is_alive():
            raise RuntimeError("A capture thread is already running. Stop it before starting a new one.")

        self.capture_thread = threading.Thread(
            target=self._capture_csi,
            args=(duration, remote_directory, exp_name, delete_prev_pcap),
            daemon=True,
        )
        self.capture_thread.start()

    def stop_csi_capture(self):
        if self.sending:
            if self.debug:
                print("Manually stopping CSI capture...")
            kill_cmd = (
                "killall rawperf"
                if self.script_category.strip().lower() == "transmitter"
                else "killall tcpdump"
            )
            self.send_command(kill_cmd, sleep=1)
            self.sending = False

    def retrieve_files(self, local_directory, remote_directory="/mnt/CSI_USB", use_ftp=False):
        os.makedirs(local_directory, exist_ok=True)
        all_files = self.send_command(f"ls {remote_directory}")
        pcap_files = [f.strip() for f in all_files if f.endswith(".pcap\n")]

        for pcap_file in pcap_files:
            remote_path = f"{remote_directory}/{pcap_file}"
            local_path = os.path.join(local_directory, pcap_file)
            self.transfer_file(remote_path[4:], local_path, use_ftp=use_ftp)

    def read_packets_count(self, full_remote_path=None, MACs=None):
        def count_elements(lst):
            lst = [item.strip() for item in lst]
            element_count = {}
            for element in lst:
                if element in element_count:
                    element_count[element] += 1
                else:
                    element_count[element] = 1
            return element_count

        tcpdump_cmd_read = (
            f"nohup /jffs/tcpdump -nn -e -r {full_remote_path}" + "| awk '{print $2}' | sort"
        )
        captured_packets = self.send_command(tcpdump_cmd_read, sleep=0)
        counts = count_elements(captured_packets)
        if self.debug:
            if MACs:
                for mac_adr in MACs:
                    if mac_adr in counts:
                        print(counts[mac_adr])
                    else:
                        print(f"{mac_adr}: 0")
            else:
                print(counts)
            print("=======================")
        return counts

    def close(self):
        if self.ssh:
            self.ssh.close()
        if self.debug:
            print("SSH connection closed.")


class WiFiCSIManager:
    """Coordinate Wi-Fi CSI capture flows across the application."""

    def __init__(self, wifi_profile: dict | None = None, *, default_remote_dir: str = DEFAULT_REMOTE_DIR):
        self.wifi_profile = wifi_profile or {}
        self.default_remote_dir = default_remote_dir.rstrip("/") + "/"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def parse_mac_addresses(mac_text: str) -> list[str]:
        if not mac_text:
            return []
        separators = [",", ";", "\n"]
        for sep in separators:
            mac_text = mac_text.replace(sep, ",")
        return [mac.strip() for mac in mac_text.split(",") if mac.strip()]

    @staticmethod
    def normalize_macs(macs: Iterable[str]) -> list[str]:
        return [mac.strip().lower() for mac in macs if mac.strip()]

    @staticmethod
    def format_channel_bandwidth(channel: str, bandwidth: str) -> str:
        channel = (channel or "").strip()
        bandwidth_value = (bandwidth or "").replace("MHz", "").strip()
        return f"{channel}/{bandwidth_value}" if bandwidth_value else channel

    @staticmethod
    def sanitize_ap_name(name: str) -> str:
        cleaned = name.strip().replace(" ", "_")
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cleaned)

    def build_ap_title(self, ap: dict) -> str:
        name = ap.get("name") or ap.get("ssid") or "Access Point"
        router_ip = ap.get("router_ssh_ip", "") or "N/A"
        frequency = ap.get("frequency", "") or "N/A"
        channel = ap.get("channel", "") or "N/A"
        bandwidth = ap.get("bandwidth", "") or "N/A"
        macs = ", ".join(self.parse_mac_addresses(ap.get("transmitter_macs", ""))) or "N/A"
        details = " | ".join([router_ip, frequency, channel, bandwidth, macs])
        return f"{details} — {name}"

    def get_capture_remote_dir(
        self,
        ap: dict,
        *,
        wifi_profile: dict | None = None,
        init_key: str = "init_test_save_directory",
    ) -> str:
        profile = wifi_profile or self.wifi_profile
        directory = ap.get(init_key) or profile.get(init_key, self.default_remote_dir)
        directory = directory or self.default_remote_dir
        return directory.rstrip("/") + "/"

    @staticmethod
    def latest_pcap_filename(candidates: list[str]) -> str | None:
        pcap_files = [f for f in candidates if f.endswith(".pcap")]
        return sorted(pcap_files)[-1] if pcap_files else None

    @staticmethod
    def filter_matching_pcaps(pcap_files: list[str], prefix: str) -> list[str]:
        return [f for f in pcap_files if f.endswith(".pcap") and f.startswith(f"{prefix}_")]

    # ------------------------------------------------------------------
    # Access point helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_sniffer(ap: dict) -> bool:
        ap_type = str(ap.get("type") or SCRIPT_CATEGORY).strip().lower()
        return ap_type != "transmitter"

    # ------------------------------------------------------------------
    # Access point ordering helpers
    # ------------------------------------------------------------------
    @classmethod
    def prioritize_transmitters_first(
        cls, access_points: Iterable[dict]
    ) -> list[dict]:
        """Return access points ordered with transmitters ahead of sniffers."""

        def _priority(ap: dict) -> tuple[int]:
            # Transmitters (rawperf) should start and stop before sniffers (tcpdump).
            # Sorting by a tuple keeps the original order among peers while grouping
            # all transmitters first.
            return (1 if cls.is_sniffer(ap) else 0,)

        return sorted(list(access_points), key=_priority)

    # ------------------------------------------------------------------
    # Router operations
    # ------------------------------------------------------------------
    @staticmethod
    def connect_router(ap: dict, router_cls=WiFiRouter, *, ssh_timeout: float | None = None):
        script_profile = (ap.get("framework") or SCRIPT_PROFILE).strip() or SCRIPT_PROFILE
        script_category = (ap.get("type") or SCRIPT_CATEGORY).strip() or SCRIPT_CATEGORY
        return router_cls(
            router_ip=ap.get("router_ssh_ip", ""),
            router_username=ap.get("router_ssh_username", ""),
            router_password=ap.get("router_ssh_password", ""),
            key_filename=ap.get("ssh_key_address", "") or None,
            debug=True,
            ssh_timeout=ssh_timeout,
            script_profile=script_profile,
            script_category=script_category,
        )

    @staticmethod
    def _is_router_connected(router) -> bool:
        transport = None
        try:
            ssh = getattr(router, "ssh", None)
            if ssh is not None:
                transport = ssh.get_transport()
        except Exception:
            return False
        return bool(transport and transport.is_active())

    def _ensure_router_connection(self, router):
        if self._is_router_connected(router):
            return

        init_method = getattr(router, "_init_ssh_connection", None)
        if callable(init_method):
            init_method()

    def ensure_router_ready(
        self,
        router,
        *,
        channel_bw: str,
        mac_addresses: list[str],
        skip_setup: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        if not skip_setup:
            setup_output = router.setup_router()
            if log and setup_output is not None:
                log(str(setup_output))

        if not getattr(router, "interface", None):
            monitor_output = router.configure_monitor_mode(
                channel=channel_bw, core=15, mac_addresses=mac_addresses
            )
            if log and monitor_output is not None:
                log(str(monitor_output))

    @staticmethod
    def start_csi_capture(router, *, duration: float, remote_directory: str, exp_name: str, delete_prev_pcap: bool):
        router.start_csi_capture_in_thread(
            duration=duration,
            remote_directory=remote_directory,
            exp_name=exp_name,
            delete_prev_pcap=delete_prev_pcap,
        )
        if getattr(router, "capture_thread", None):
            router.capture_thread.join()

    def download_capture(
        self,
        router,
        *,
        remote_dir: str,
        target_file: str,
        local_directory: Path,
        use_ftp: bool,
        progress_cb=None,
    ) -> Tuple[Path, str]:
        local_directory.mkdir(parents=True, exist_ok=True)
        local_path = local_directory / target_file
        remote_path = f"{remote_dir.rstrip('/')}/{target_file}"
        self._transfer_file(
            router,
            remote_path=remote_path,
            local_path=str(local_path),
            use_ftp=use_ftp,
            progress_cb=progress_cb,
        )
        return local_path, remote_path

    def _transfer_file(
        self,
        router,
        *,
        remote_path: str,
        local_path: str,
        use_ftp: bool,
        progress_cb=None,
    ):
        if use_ftp or progress_cb is None:
            router.transfer_file(remote_path[4:], local_path, use_ftp=use_ftp)
            if progress_cb is not None:
                try:
                    progress_cb(remote_path, 0, 0)
                except Exception:
                    pass
            return

        transport = getattr(router, "ssh", None)
        if transport is None:
            router.transfer_file(remote_path[4:], local_path, use_ftp=use_ftp)
            return

        def _progress(filename, size, sent):
            try:
                progress_cb(filename, int(size), int(sent))
            except Exception:
                pass

        with SCPClient(transport.get_transport(), progress=_progress) as scp:
            scp.get(remote_path, local_path)

    def reboot_access_points(
        self,
        access_points: list[dict],
        *,
        router_cls=WiFiRouter,
        existing_routers: dict[str, object] | None = None,
        log: Callable[[str], None] | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
    ):
        routers = existing_routers or {}

        for idx, ap in enumerate(access_points, start=1):
            ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx}"
            ap_key = self.sanitize_ap_name(ap_name)
            if progress_cb:
                progress_cb(idx - 1, f"Rebooting {ap_name}…")

            router = routers.get(ap_key)

            if router is None:
                try:
                    router = self.connect_router(ap, router_cls)
                except Exception as exc:  # pragma: no cover - network dependency
                    if log:
                        log(f"Failed to connect to {ap_name}: {exc}")
                    if progress_cb:
                        progress_cb(idx, f"Failed to reboot {ap_name}.")
                    continue

            try:
                self._ensure_router_connection(router)
                router.send_command("/sbin/reboot -f > /dev/null 2>&1 &")
                if log:
                    log(f"Sent reboot command to {ap_name}.")
            except Exception as exc:  # pragma: no cover - network dependency
                if log:
                    log(f"Failed to reboot {ap_name}: {exc}")
            finally:
                if router is not None:
                    try:
                        router.close()
                    except Exception:
                        pass

            if progress_cb:
                progress_cb(idx, f"Rebooted {ap_name}.")

        if progress_cb:
            progress_cb(len(access_points), "All access points rebooted.")

    def reboot_access_points_with_dialog(
        self,
        access_points: list[dict],
        *,
        router_cls=WiFiRouter,
        parent=None,
        existing_routers: dict[str, object] | None = None,
        log: Callable[[str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        thread_name: str = "AccessPointRebooter",
    ) -> tuple[QMessageBox, threading.Thread]:
        dialog = QMessageBox(parent)
        dialog.setWindowTitle("WIRLab - Rebooting Access Points")
        dialog.setText("Sending reboot commands to access points…")
        dialog.setIcon(QMessageBox.Information)
        dialog.setStandardButtons(QMessageBox.Close)
        dialog.setDefaultButton(QMessageBox.Close)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.show()

        def _update_progress(message: str):
            try:
                dialog.setInformativeText(message)
            except Exception:
                pass

        def progress_cb(_: int, message: str):
            QTimer.singleShot(0, lambda m=message: _update_progress(m))

        def _finish_dialog():
            try:
                dialog.setText("Reboot commands executed for all access points.")
                dialog.setInformativeText("You can close this window once you are ready.")
                dialog.show()
            except Exception:
                pass

            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    pass

        def worker():
            try:
                self.reboot_access_points(
                    access_points,
                    router_cls=router_cls,
                    existing_routers=existing_routers,
                    log=log,
                    progress_cb=progress_cb,
                )
            finally:
                QTimer.singleShot(0, _finish_dialog)

        thread = threading.Thread(
            target=worker,
            name=thread_name,
            daemon=True,
        )
        thread.start()
        return dialog, thread

    def summarize_packets(self, capture_path: Path, mac_addresses: list[str]) -> tuple[int, dict[str, int]]:
        normalized_macs = self.normalize_macs(mac_addresses)
        try:
            return count_packets_for_macs(str(capture_path), normalized_macs)
        except Exception:  # pragma: no cover - CSIKit dependency handling
            return 0, {mac: 0 for mac in normalized_macs}
