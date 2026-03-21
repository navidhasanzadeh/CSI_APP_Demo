"""Dialogs and workers for Wi-Fi CSI capture and hardware initialization."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from ap_workflow_dialog import BaseAccessPointDialog
from wifi_csi_manager import DEFAULT_REMOTE_DIR, WiFiCSIManager, WiFiRouter
import time_reference as time_reference_module

CSI_MANAGER = WiFiCSIManager()


class CSICaptureWorker(QObject):
    status_changed = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(bool, list)

    def __init__(
        self,
        access_points: list[dict],
        *,
        duration: float,
        wifi_profile: dict,
        build_prefix: Callable[[str], str],
    ):
        super().__init__()
        self.access_points = access_points
        self.duration = max(float(duration), 1.0)
        self.wifi_profile = wifi_profile
        self.build_prefix = build_prefix
        self._routers_info: list[dict] = []
        self._stop_event = threading.Event()
        self._router_lock = threading.Lock()
        self._router = None
        scenario_value = str(self.wifi_profile.get("csi_capture_scenario", "scenario_2"))
        delete_prev_pcap = bool(self.wifi_profile.get("delete_prev_pcap", False))
        if scenario_value.lower() == "scenario_2":
            delete_prev_pcap = False
        self._delete_prev_pcap = delete_prev_pcap

    def request_stop(self):
        self._stop_event.set()
        with self._router_lock:
            router = self._router
            self._router = None
        if router is not None:
            try:
                category = str(getattr(router, "script_category", "")).strip().lower()
                kill_cmd = "killall rawperf" if category == "transmitter" else "killall tcpdump"
                router.send_command(kill_cmd, sleep=0)
            except Exception:
                pass
            try:
                router.close()
            except Exception:
                pass

    def run(self):
        ordered_access_points = CSI_MANAGER.prioritize_transmitters_first(
            self.access_points
        )

        for idx, ap in enumerate(ordered_access_points):
            if self._stop_event.is_set():
                self.log_message.emit("CSI capture start cancelled by user.")
                self.finished.emit(False, self._routers_info)
                return

            ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx + 1}"
            self.status_changed.emit(idx, "yellow", f"Connecting to {ap_name}")
            router_ip = ap.get("router_ssh_ip", "") or "N/A"
            self.log_message.emit(
                f"Connecting to {ap_name} (router IP: {router_ip}, duration: {self.duration}s)"
            )
            skip_setup = bool(
                ap.get("initialized_success")
                or ap.get("skip_router_setup")
                or self.wifi_profile.get("skip_router_setup")
            )

            router = None
            keep_router_open = False
            try:
                router = CSI_MANAGER.connect_router(ap, WiFiRouter)
                with self._router_lock:
                    self._router = router
                if self._stop_event.is_set():
                    self.finished.emit(False, self._routers_info)
                    return

                mac_addresses = CSI_MANAGER.parse_mac_addresses(
                    ap.get("transmitter_macs", "")
                )
                channel_bw = CSI_MANAGER.format_channel_bandwidth(
                    ap.get("channel", ""), ap.get("bandwidth", "")
                )
                if not skip_setup:
                    self.log_message.emit(
                        "Configuring monitor mode with channel/bandwidth "
                        f"{channel_bw} and MACs: {mac_addresses}"
                    )
                CSI_MANAGER.ensure_router_ready(
                    router,
                    channel_bw=channel_bw,
                    mac_addresses=mac_addresses,
                    log=self.log_message.emit,
                    skip_setup=skip_setup,
                )

                exp_name = self.build_prefix(ap_name)
                remote_dir = CSI_MANAGER.get_capture_remote_dir(ap, wifi_profile=self.wifi_profile)
                self.log_message.emit(
                    "Router configured; deferring CSI capture start until main window is ready."
                )
                self.status_changed.emit(idx, "green", "Ready for capture")
                ap_key = CSI_MANAGER.sanitize_ap_name(ap_name)
                self._routers_info.append(
                    {
                        "key": ap_key,
                        "router": router,
                        "run_info": {
                            "ap": ap,
                            "exp_name": exp_name,
                            "remote_dir": remote_dir,
                            "macs": mac_addresses,
                            "channel_bw": channel_bw,
                            "duration": self.duration,
                            "delete_prev_pcap": self._delete_prev_pcap,
                            "started": False,
                        },
                    }
                )
                router = None  # ownership transferred
                keep_router_open = True
            except Exception as exc:  # pragma: no cover - network dependency
                self.status_changed.emit(idx, "red", str(exc))
                self.log_message.emit(f"Failed to start capture on {ap_name}: {exc}")
                self.error.emit(f"Failed to start capture on {ap_name}: {exc}")
                self.finished.emit(False, self._routers_info)
                return
            finally:
                with self._router_lock:
                    active_router = self._router
                    self._router = None
                if active_router is None:
                    active_router = router
                if active_router is not None and not keep_router_open:
                    try:
                        active_router.close()
                    except Exception:
                        pass

        self.finished.emit(True, self._routers_info)


class CSICaptureStartupDialog(BaseAccessPointDialog):
    error_title = "CSI Capture Failed"

    def __init__(
        self,
        profile_name: str,
        access_points: list[dict],
        *,
        duration: float,
        wifi_profile: dict,
        build_prefix: Callable[[str], str],
        parent=None,
        log_file_path: str | Path | None = None,
    ):
        self.routers_info: list[dict] = []
        self.successful = False
        self.duration = duration
        self.wifi_profile = wifi_profile
        self.build_prefix = build_prefix
        self._log_file_path = log_file_path

        super().__init__(
            profile_name=profile_name,
            access_points=access_points,
            heading_template="Starting CSI collection for Wi-Fi profile: {profile_name}",
            window_title="Start CSI Capture",
            start_message="Starting CSI capture...",
            parent=parent,
            log_file_path=log_file_path,
        )

    def _create_worker(self):
        return CSICaptureWorker(
            self.sorted_access_points,
            duration=self.duration,
            wifi_profile=self.wifi_profile,
            build_prefix=self.build_prefix,
        )

    def _on_error(self, message: str):
        super()._on_error(message)
        self.reject()

    def _handle_finished(self, success: bool, routers_info: list):
        self.successful = success
        self.routers_info = routers_info or []
        if success:
            self._append_log("CSI capture started for all access points.")
            self.accept()
        else:
            self._append_log("CSI capture failed or was cancelled.")
            self.reject()


class HardwareInitWorker(QObject):
    status_changed = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(bool, list)

    def __init__(
        self,
        access_points: list[dict],
        *,
        test_duration: float,
        save_directory: str,
        local_directory: Path,
    ):
        super().__init__()
        self.access_points = access_points
        self._stop_event = threading.Event()
        self._router_lock = threading.Lock()
        self._router = None
        self.test_duration = float(test_duration)
        self.default_save_directory = self._normalize_remote_dir(save_directory)
        self.local_directory = local_directory
        self.downloaded_files: list[dict] = []

    @staticmethod
    def _normalize_remote_dir(save_directory: str) -> str:
        normalized = (save_directory or DEFAULT_REMOTE_DIR).rstrip("/")
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def request_stop(self):
        self._stop_event.set()
        self._stop_current_router()

    def _stop_current_router(self):
        with self._router_lock:
            router = self._router
            self._router = None

        if router is None:
            return

        try:
            category = str(getattr(router, "script_category", "")).strip().lower()
            kill_cmd = "killall rawperf" if category == "transmitter" else "killall tcpdump"
            router.send_command(kill_cmd, sleep=1)
        except Exception:
            pass

        try:
            router.close()
        except Exception:
            pass

    def run(self):
        ordered_access_points = CSI_MANAGER.prioritize_transmitters_first(
            self.access_points
        )

        for idx, ap in enumerate(ordered_access_points):
            if self._stop_event.is_set():
                self.log_message.emit("Initialization cancelled by user.")
                self.finished.emit(False, self.downloaded_files)
                return

            ap_name = ap.get("name") or f"Access Point {idx + 1}"
            is_sniffer = CSI_MANAGER.is_sniffer(ap)
            self.status_changed.emit(idx, "yellow", f"Connecting to {ap_name}")
            self.log_message.emit(f"Starting initialization for {ap_name}")

            router = None
            try:
                router = CSI_MANAGER.connect_router(ap, WiFiRouter)
                with self._router_lock:
                    self._router = router
                if self._stop_event.is_set():
                    self.log_message.emit("Initialization cancelled by user.")
                    self.finished.emit(False, self.downloaded_files)
                    return

                self.log_message.emit("Running setup_router()")
                setup_output = router.setup_router()
                if setup_output is not None:
                    self.log_message.emit(str(setup_output))

                channel_bw = CSI_MANAGER.format_channel_bandwidth(
                    ap.get("channel", ""), ap.get("bandwidth", "")
                )
                mac_addresses = CSI_MANAGER.parse_mac_addresses(ap.get("transmitter_macs", ""))
                self.log_message.emit(
                    f"Configuring monitor mode with channel {channel_bw} "
                    f"and MACs: {mac_addresses}"
                )
                CSI_MANAGER.ensure_router_ready(
                    router,
                    channel_bw=channel_bw,
                    mac_addresses=mac_addresses,
                    log=self.log_message.emit,
                )

                sanitized_name = CSI_MANAGER.sanitize_ap_name(ap_name)
                exp_name = f"init_test_{sanitized_name}"
                remote_dir = self._normalize_remote_dir(
                    ap.get("init_test_save_directory") or self.default_save_directory
                )
                self.log_message.emit(
                    f"Starting init CSI capture for {ap_name}: duration {self.test_duration}s, "
                    f"remote directory '{remote_dir}', exp name '{exp_name}'."
                )
                CSI_MANAGER.start_csi_capture(
                    router,
                    duration=self.test_duration,
                    remote_directory=remote_dir,
                    exp_name=exp_name,
                    delete_prev_pcap=False,
                )

                if not is_sniffer:
                    self.log_message.emit(
                        f"{ap_name} identified as transmitter; skipping CSI file transfer."
                    )
                    continue

                remote_listing = router.send_command(f"ls {remote_dir}")
                pcap_files = [line.strip() for line in remote_listing]
                matching_files = CSI_MANAGER.filter_matching_pcaps(pcap_files, exp_name)
                target_file = CSI_MANAGER.latest_pcap_filename(matching_files)
                if not target_file:
                    raise RuntimeError(
                        f"No CSI capture file matching '{exp_name}_*.pcap' found on router."
                    )

                local_directory = self.local_directory
                local_directory.mkdir(parents=True, exist_ok=True)
                local_path = local_directory / target_file
                use_ftp = str(ap.get("download_mode", "SFTP")).strip().upper() != "SFTP"
                local_path, _ = CSI_MANAGER.download_capture(
                    router,
                    remote_dir=remote_dir,
                    target_file=target_file,
                    local_directory=local_directory,
                    use_ftp=use_ftp,
                )

                if not local_path.exists():
                    raise RuntimeError(f"CSI capture file '{target_file}' failed to download.")
                size_kb = local_path.stat().st_size / 1024
                total_packets, mac_packet_counts = CSI_MANAGER.summarize_packets(
                    local_path, mac_addresses
                )
                self.downloaded_files.append(
                    {
                        "ap_name": ap_name,
                        "router_ip": ap.get("router_ssh_ip", "N/A"),
                        "file_name": target_file,
                        "local_path": local_path,
                        "size_kb": size_kb,
                        "total_packets": total_packets,
                        "mac_packet_counts": mac_packet_counts,
                        "transmitter_macs": mac_addresses,
                        "duration": self.test_duration,
                    }
                )
                self.log_message.emit(
                    f"Downloaded init CSI capture for {ap_name} to {local_path} "
                    f"({size_kb:.1f} KB)."
                )
                mac_summary = ", ".join(
                    f"{mac}:{mac_packet_counts.get(mac.lower(), 0)}"
                    for mac in CSI_MANAGER.normalize_macs(mac_addresses)
                )
                self.log_message.emit(
                    f"Packet summary for {ap_name} — total: {total_packets}, "
                    f"transmitters: {mac_summary or 'N/A'}"
                )
            except Exception as exc:  # pragma: no cover - network operations
                self.status_changed.emit(idx, "red", str(exc))
                self.error.emit(f"Failed to initialize {ap_name}: {exc}")
                self.finished.emit(False, self.downloaded_files)
                return
            finally:
                with self._router_lock:
                    active_router = self._router
                    self._router = None
                if active_router is None:
                    active_router = router
                if active_router:
                    try:
                        active_router.close()
                    except Exception:
                        pass

            self.status_changed.emit(idx, "green", "Completed")
            self.log_message.emit(f"Finished initialization for {ap_name}")

        self.finished.emit(True, self.downloaded_files)


class HardwareInitializationDialog(BaseAccessPointDialog):
    error_title = "Hardware Initialization Failed"

    def __init__(
        self,
        profile_name: str,
        access_points: list[dict],
        *,
        test_duration: float,
        save_directory: str,
        parent=None,
    ):
        self.test_duration = test_duration
        self.save_directory = save_directory
        timestamp = (
            time_reference_module.get_global_time_reference()
            .now_datetime()
            .strftime("%Y%m%d_%H%M%S")
        )
        self.run_directory = Path("results") / "InitTest" / timestamp
        self.log_file_path = self.run_directory / f"hardware_init_{timestamp}.log"
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.was_successful = False

        super().__init__(
            profile_name=profile_name,
            access_points=access_points,
            heading_template="Initializing hardware for Wi-Fi profile: {profile_name}",
            window_title="Initialize Hardwares",
            start_message="Starting hardware initialization...",
            parent=parent,
            log_file_path=self.log_file_path,
        )

    def _create_worker(self):
        return HardwareInitWorker(
            self.sorted_access_points,
            test_duration=self.test_duration,
            save_directory=self.save_directory,
            local_directory=self.run_directory,
        )

    def _format_summary_table(self, downloaded_files: list[dict]) -> str:
        headers = [
            "Access Point",
            "Router SSH IP",
            "Total Packets",
            "Transmitter MAC",
            "MAC Packets",
            "Rate (pkt/s)",
        ]
        rows: list[list[str]] = []
        for item in downloaded_files:
            mac_counts = item.get("mac_packet_counts") or {}
            macs = CSI_MANAGER.normalize_macs(item.get("transmitter_macs", [])) or ["N/A"]
            total_packets = item.get("total_packets")
            duration = float(item.get("duration") or 0) or 0.0
            for mac in macs:
                mac_packets = mac_counts.get(mac, 0)
                rate = mac_packets / duration if duration else 0.0
                rows.append(
                    [
                        item.get("ap_name", "N/A"),
                        str(item.get("router_ip", "N/A")),
                        str(total_packets if total_packets is not None else "N/A"),
                        mac,
                        str(mac_packets),
                        f"{rate:.2f}",
                    ]
                )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, value in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(value))

        def _format_row(values: list[str]) -> str:
            return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

        divider = "-+-".join("-" * width for width in col_widths)
        table_lines = [_format_row(headers), divider]
        table_lines.extend(_format_row(row) for row in rows)
        return "\n".join(table_lines)

    def _handle_finished(self, success: bool, downloaded_files: list):
        self.was_successful = success
        if success:
            self._append_log("Hardware initialization completed successfully.")
            if downloaded_files:
                for item in downloaded_files:
                    self._append_log(
                        f"Downloaded {item.get('file_name')} ({item.get('size_kb', 0):.1f} KB) "
                        f"for {item.get('ap_name')}"
                    )
                message = "All access points initialized successfully.\n\n"
                message += self._format_summary_table(downloaded_files)
                message += "\n\nDownloaded files:\n"
                message += "\n".join(
                    f"- {item.get('file_name')} ({item.get('size_kb', 0):.1f} KB)"
                    for item in downloaded_files
                )
            else:
                message = "All access points initialized successfully."
            QMessageBox.information(self, "Initialization Complete", message)
        else:
            self._append_log("Hardware initialization stopped due to an error or cancellation.")


class RouterConnectionTestWorker(QObject):
    status_changed = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(bool, list)

    def __init__(self, access_points: list[dict], *, wifi_profile: dict):
        super().__init__()
        self.access_points = access_points
        self.wifi_profile = wifi_profile or {}
        self.wifi_manager = WiFiCSIManager(wifi_profile=self.wifi_profile)
        self._stop_event = threading.Event()
        self._router_lock = threading.Lock()
        self._router = None

    def request_stop(self):
        self._stop_event.set()
        with self._router_lock:
            router = self._router
            self._router = None
        if router is not None:
            try:
                router.close()
            except Exception:
                pass

    def _check_uname(self, router) -> tuple[bool, str]:
        uname_output = router.send_command("uname")
        output_text = "".join(uname_output).strip()
        linux_ok = any("linux" in (line or "").strip().lower() for line in uname_output)
        return linux_ok, output_text or "(no output)"

    def _check_remote_directory(self, router, ap: dict) -> tuple[bool, str]:
        remote_dir = self.wifi_manager.get_capture_remote_dir(
            ap, wifi_profile=self.wifi_profile
        ).rstrip("/")
        check_command = (
            f'[ -d "{remote_dir}" ] && echo "__DIR_OK__" '  # noqa: ISC003
            f'|| echo "__DIR_MISSING__"'
        )
        dir_output = router.send_command(check_command)
        directory_ok = any("__DIR_OK__" in (line or "") for line in dir_output)
        output_text = "".join(dir_output).strip()
        return directory_ok, output_text or "(no output)"

    def run(self):
        success = True
        results: list[dict] = []

        for idx, ap in enumerate(self.access_points):
            if self._stop_event.is_set():
                self.log_message.emit("Connection test cancelled by user.")
                self.finished.emit(False, results)
                return

            ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx + 1}"
            router_ip = ap.get("router_ssh_ip", "") or "N/A"
            self.status_changed.emit(idx, "yellow", f"Connecting to {ap_name}")
            self.log_message.emit(f"Connecting to {ap_name} (router IP: {router_ip})")

            router = None
            try:
                router = self.wifi_manager.connect_router(
                    ap, WiFiRouter, ssh_timeout=2.0
                )
                with self._router_lock:
                    self._router = router

                if self._stop_event.is_set():
                    self.log_message.emit("Connection test cancelled by user.")
                    self.finished.emit(False, results)
                    return

                linux_ok, uname_output = self._check_uname(router)
                self.log_message.emit(f"uname output for {ap_name}: {uname_output}")

                directory_ok, dir_output = self._check_remote_directory(router, ap)
                self.log_message.emit(
                    f"Remote directory check for {ap_name}: {dir_output}"
                )

                all_ok = linux_ok and directory_ok
                success = success and all_ok

                tooltip_parts = []
                tooltip_parts.append("Linux detected" if linux_ok else "Unexpected uname output")
                tooltip_parts.append(
                    "Directory exists" if directory_ok else "Directory missing"
                )
                tooltip = "; ".join(tooltip_parts)
                status_color = "green" if all_ok else "red"
                self.status_changed.emit(idx, status_color, tooltip)

                results.append(
                    {
                        "ap_name": ap_name,
                        "router_ip": router_ip,
                        "linux_ok": linux_ok,
                        "directory_ok": directory_ok,
                        "uname_output": uname_output,
                        "directory_output": dir_output,
                    }
                )
            except Exception as exc:  # pragma: no cover - network dependency
                success = False
                message = f"Failed connection test for {ap_name}: {exc}"
                self.status_changed.emit(idx, "red", str(exc))
                self.log_message.emit(message)
                results.append(
                    {
                        "ap_name": ap_name,
                        "router_ip": router_ip,
                        "linux_ok": False,
                        "directory_ok": False,
                        "error": str(exc),
                    }
                )
            finally:
                with self._router_lock:
                    active_router = self._router
                    self._router = None
                if active_router is None:
                    active_router = router
                if active_router is not None:
                    try:
                        active_router.close()
                    except Exception:
                        pass

        self.finished.emit(success, results)


class RouterPcapCleanupWorker(QObject):
    status_changed = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(bool, list)

    def __init__(self, access_points: list[dict], *, wifi_profile: dict):
        super().__init__()
        self.access_points = access_points
        self.wifi_profile = wifi_profile or {}
        self.wifi_manager = WiFiCSIManager(wifi_profile=self.wifi_profile)
        self._stop_event = threading.Event()
        self._router_lock = threading.Lock()
        self._router = None

    def request_stop(self):
        self._stop_event.set()
        with self._router_lock:
            router = self._router
            self._router = None
        if router is not None:
            try:
                router.close()
            except Exception:
                pass

    def _count_pcaps(self, router, remote_dir: str) -> int:
        command = (
            f"sh -c 'ls -1 \"{remote_dir}\"/*.pcap 2>/dev/null | wc -l'"
        )
        output = router.send_command(command)
        if not output:
            return 0
        try:
            return int(str(output[-1]).strip())
        except ValueError:
            return 0

    def _delete_pcaps(self, router, remote_dir: str):
        delete_command = f"rm -f \"{remote_dir}\"/*.pcap"
        router.send_command(delete_command)

    def run(self):
        success = True
        results: list[dict] = []

        for idx, ap in enumerate(self.access_points):
            if self._stop_event.is_set():
                self.log_message.emit("PCAP cleanup cancelled by user.")
                self.finished.emit(False, results)
                return

            ap_name = ap.get("name") or ap.get("ssid") or f"Access Point {idx + 1}"
            router_ip = ap.get("router_ssh_ip", "") or "N/A"
            self.status_changed.emit(idx, "yellow", f"Connecting to {ap_name}")
            self.log_message.emit(
                f"Connecting to {ap_name} (router IP: {router_ip})"
            )

            router = None
            remote_dir = self.wifi_manager.get_capture_remote_dir(
                ap, wifi_profile=self.wifi_profile
            ).rstrip("/")
            try:
                router = self.wifi_manager.connect_router(
                    ap, WiFiRouter, ssh_timeout=2.0
                )
                with self._router_lock:
                    self._router = router

                if self._stop_event.is_set():
                    self.log_message.emit("PCAP cleanup cancelled by user.")
                    self.finished.emit(False, results)
                    return

                self.log_message.emit(
                    f"Counting PCAP files in {remote_dir} for {ap_name}."
                )
                pcap_count = self._count_pcaps(router, remote_dir)
                self.log_message.emit(
                    f"Deleting {pcap_count} PCAP file(s) from {remote_dir}."
                )
                self._delete_pcaps(router, remote_dir)

                self.status_changed.emit(
                    idx,
                    "green",
                    f"Deleted {pcap_count} PCAP file(s)",
                )
                results.append(
                    {
                        "ap_name": ap_name,
                        "router_ip": router_ip,
                        "remote_dir": remote_dir,
                        "deleted_count": pcap_count,
                    }
                )
            except Exception as exc:  # pragma: no cover - network dependency
                success = False
                message = f"Failed PCAP cleanup for {ap_name}: {exc}"
                self.status_changed.emit(idx, "red", str(exc))
                self.log_message.emit(message)
                results.append(
                    {
                        "ap_name": ap_name,
                        "router_ip": router_ip,
                        "remote_dir": remote_dir,
                        "deleted_count": 0,
                        "error": str(exc),
                    }
                )
            finally:
                with self._router_lock:
                    active_router = self._router
                    self._router = None
                if active_router is None:
                    active_router = router
                if active_router is not None:
                    try:
                        active_router.close()
                    except Exception:
                        pass

        self.finished.emit(success, results)


class RouterConnectionTestDialog(BaseAccessPointDialog):
    error_title = "Connection Test Failed"

    def __init__(
        self,
        profile_name: str,
        access_points: list[dict],
        *,
        wifi_profile: dict,
        parent=None,
    ):
        self.wifi_profile = wifi_profile or {}
        timestamp = (
            time_reference_module.get_global_time_reference()
            .now_datetime()
            .strftime("%Y%m%d_%H%M%S")
        )
        self.run_directory = Path("results") / "ConnectionTests" / timestamp
        self.log_file_path = self.run_directory / f"connection_test_{timestamp}.log"
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.was_successful = False
        self.results: list[dict] = []

        super().__init__(
            profile_name=profile_name,
            access_points=access_points,
            heading_template="Testing router connections for Wi-Fi profile: {profile_name}",
            window_title="Test Router Connections",
            start_message="Starting connection tests...",
            parent=parent,
            log_file_path=self.log_file_path,
        )

    def _create_worker(self):
        return RouterConnectionTestWorker(
            self.sorted_access_points, wifi_profile=self.wifi_profile
        )

    def _handle_finished(self, success: bool, results: list[dict]):
        self.was_successful = success
        self.results = results or []

        failed = [
            result
            for result in self.results
            if not (result.get("linux_ok") and result.get("directory_ok"))
        ]

        if success and not failed:
            self._append_log(
                "All routers responded with Linux and the configured directories are reachable."
            )
            QMessageBox.information(
                self,
                "Connection Test Complete",
                "All routers responded with Linux and the directories exist.",
            )
        else:
            self._append_log(
                "Connection test completed with issues. Review the details above."
            )
            summary_lines = []
            for item in failed:
                ap_name = item.get("ap_name") or "Unknown AP"
                linux_status = "OK" if item.get("linux_ok") else "Failed"
                dir_status = "OK" if item.get("directory_ok") else "Missing"
                summary_lines.append(
                    f"- {ap_name} (Linux: {linux_status}, Directory: {dir_status})"
                )
            summary = "\n".join(summary_lines) if summary_lines else "Unknown errors"
            QMessageBox.warning(
                self,
                "Connection Test Issues",
                "Some routers failed the checks:\n" + summary,
            )


class RouterPcapCleanupDialog(BaseAccessPointDialog):
    error_title = "PCAP Cleanup Failed"

    def __init__(
        self,
        profile_name: str,
        access_points: list[dict],
        *,
        wifi_profile: dict,
        parent=None,
    ):
        self.wifi_profile = wifi_profile or {}
        timestamp = (
            time_reference_module.get_global_time_reference()
            .now_datetime()
            .strftime("%Y%m%d_%H%M%S")
        )
        self.run_directory = Path("results") / "PcapCleanup" / timestamp
        self.log_file_path = self.run_directory / f"pcap_cleanup_{timestamp}.log"
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.was_successful = False
        self.results: list[dict] = []

        super().__init__(
            profile_name=profile_name,
            access_points=access_points,
            heading_template="Cleaning PCAP files for Wi-Fi profile: {profile_name}",
            window_title="Delete Router PCAP Files",
            start_message="Starting PCAP cleanup...",
            parent=parent,
            log_file_path=self.log_file_path,
        )

    def _create_worker(self):
        return RouterPcapCleanupWorker(
            self.sorted_access_points, wifi_profile=self.wifi_profile
        )

    def _handle_finished(self, success: bool, results: list[dict]):
        self.was_successful = success
        self.results = results or []

        summary_lines = []
        for item in self.results:
            ap_name = item.get("ap_name") or "Unknown AP"
            deleted = item.get("deleted_count", 0)
            if item.get("error"):
                summary_lines.append(f"- {ap_name}: Error ({item.get('error')})")
            else:
                summary_lines.append(f"- {ap_name}: Deleted {deleted} PCAP file(s)")

        summary = "\n".join(summary_lines) if summary_lines else "No routers processed."
        if success:
            self._append_log("PCAP cleanup completed successfully.")
            QMessageBox.information(
                self,
                "PCAP Cleanup Complete",
                "PCAP cleanup completed:\n" + summary,
            )
        else:
            self._append_log("PCAP cleanup completed with errors.")
            QMessageBox.warning(
                self,
                "PCAP Cleanup Issues",
                "PCAP cleanup completed with errors:\n" + summary,
            )
