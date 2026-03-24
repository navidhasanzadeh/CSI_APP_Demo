import datetime
import sys
import argparse
from copy import deepcopy

from PyQt5.QtWidgets import QApplication, QDialog

from MainWindow import MainWindow
from demo_window import DemoWindow
from csi_capture_window import CSICaptureStartupDialog
from config_window import (
    load_participant_profiles,
    load_experiment_profiles,
    load_environment_profiles,
    load_action_profiles,
    load_ui_profiles,
    load_voice_profiles,
    load_camera_profiles,
    load_depth_camera_profiles,
    load_wifi_profiles,
    load_demo_profiles,
    load_time_profiles,
    save_participant_profiles,
    save_experiment_profiles,
    save_action_profiles,
    save_environment_profiles,
    save_ui_profiles,
    save_voice_profiles,
    save_camera_profiles,
    save_depth_camera_profiles,
    save_wifi_profiles,
    save_demo_profiles,
    save_time_profiles,
    save_selected_profile_choices,
    ConfigDialog,
    generate_participant_id,
    DEFAULT_ENVIRONMENT_PROFILE,
    filter_blank_actions,
)
import time_reference
from password_manager import set_password_bypass


def _sanitize_ap_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cleaned)


def _combined_value(primary: str, secondary: str, *, fallback: str) -> str:
    primary_clean = _sanitize_ap_name(primary or "")
    secondary_clean = _sanitize_ap_name(secondary or "")
    if secondary_clean:
        if primary_clean:
            return f"{primary_clean}-and-{secondary_clean}"
        return secondary_clean
    return primary_clean or fallback


def _compute_expected_duration(experiment: dict, actions: dict) -> float:
    stop_time = float(experiment.get("stop_time", 6))
    action_time = float(experiment.get("action_time", 4))
    repetitions = int(experiment.get("each_action_repitition_times", 20))
    beginning_baseline = float(experiment.get("beginning_baseline_recording", 15))
    between_baseline = float(experiment.get("between_actions_baseline_recording", 10))
    ending_baseline = float(experiment.get("ending_baseline_recording", 20))
    action_count = len(actions or {})

    total = beginning_baseline + ending_baseline
    action_block = max(0.0, stop_time + action_time)
    if action_count:
        total += action_count * repetitions * action_block
        if action_count > 1:
            total += (action_count - 1) * between_baseline
    return max(total, 0.0)


def _build_capture_prefix(subject: dict, ap_name: str, *, time_ref: time_reference.TimeReference) -> str:
    timestamp = time_ref.now_datetime().strftime("%Y%m%d_%H%M%S")
    has_second = bool(subject.get("has_second_participant"))
    participant = _combined_value(
        subject.get("name", "participant"),
        subject.get("second_name", "") if has_second else "",
        fallback="participant",
    )
    age_group = _combined_value(
        subject.get("age_group", "no_age"),
        subject.get("second_age_group", "") if has_second else "",
        fallback="no_age",
    )
    participant_id = _combined_value(
        subject.get("participant_id", "pid"),
        subject.get("second_participant_id", "") if has_second else "",
        fallback="pid",
    )
    exp_id = _sanitize_ap_name(subject.get("experiment_id", "exp"))
    sanitized_ap = _sanitize_ap_name(ap_name)
    return f"csi_{participant}_{participant_id}_{age_group}_{exp_id}_{sanitized_ap}_{timestamp}"


def main(argv: list[str] | None = None):
    argv = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(description="CSI Collection Software")
    parser.add_argument(
        "--no-password",
        "--admin",
        action="store_true",
        dest="no_password",
        help="Run in admin mode without prompting for passwords.",
    )
    args, qt_args = parser.parse_known_args(argv)

    if args.no_password:
        set_password_bypass(True)

    app = QApplication([sys.argv[0]] + qt_args)

    participant_profiles = load_participant_profiles()
    experiment_profiles = load_experiment_profiles()
    environment_profiles = load_environment_profiles()
    action_profiles = load_action_profiles()
    voice_profiles = load_voice_profiles()
    ui_profiles = load_ui_profiles()
    camera_profiles = load_camera_profiles(experiment_profiles)
    depth_camera_profiles = load_depth_camera_profiles()
    wifi_profiles = load_wifi_profiles()
    demo_profiles = load_demo_profiles()
    time_profiles = load_time_profiles()

    cfg = ConfigDialog(
        participant_profiles,
        experiment_profiles,
        environment_profiles,
        action_profiles,
        voice_profiles,
        camera_profiles,
        depth_camera_profiles,
        ui_profiles,
        wifi_profiles,
        demo_profiles,
        time_profiles,
    )
    result = cfg.exec_()

    if result != QDialog.Accepted:
        sys.exit(0)

    participant_profiles = cfg.get_participant_profiles()
    experiment_profiles = cfg.get_experiment_profiles()
    environment_profiles = cfg.get_environment_profiles()
    action_profiles = cfg.get_action_profiles()
    voice_profiles = cfg.get_voice_profiles()
    ui_profiles = cfg.get_ui_profiles()
    camera_profiles = cfg.get_camera_profiles()
    depth_camera_profiles = cfg.get_depth_camera_profiles()
    wifi_profiles = cfg.get_wifi_profiles()
    demo_profiles = cfg.get_demo_profiles()
    time_profiles = cfg.get_time_profiles()

    selected_participant = cfg.get_selected_participant_name()
    selected_experiment = cfg.get_selected_experiment_name()
    selected_environment = cfg.get_selected_environment_name()
    selected_actions = cfg.get_selected_action_profile_name()
    selected_camera = cfg.get_selected_camera_profile_name()
    selected_depth_camera = cfg.get_selected_depth_camera_profile_name()
    selected_voice = cfg.get_selected_voice_profile_name()
    selected_ui = cfg.get_selected_ui_profile_name()
    selected_wifi = cfg.get_selected_wifi_profile_name()
    selected_demo = cfg.get_selected_demo_profile_name()
    selected_time = cfg.get_selected_time_profile_name()

    save_selected_profile_choices(
        {
            "participant": selected_participant,
            "experiment": selected_experiment,
            "environment": selected_environment,
            "actions": selected_actions,
            "camera": selected_camera,
            "depth_camera": selected_depth_camera,
            "voice": selected_voice,
            "ui": selected_ui,
            "wifi": selected_wifi,
            "demo": selected_demo,
            "time": selected_time,
        }
    )

    save_participant_profiles(participant_profiles)
    save_experiment_profiles(experiment_profiles)
    save_environment_profiles(environment_profiles)
    save_action_profiles(action_profiles)
    save_ui_profiles(ui_profiles)
    save_voice_profiles(voice_profiles)
    save_camera_profiles(camera_profiles)
    save_depth_camera_profiles(depth_camera_profiles)
    save_wifi_profiles(wifi_profiles)
    save_demo_profiles(demo_profiles)
    save_time_profiles(time_profiles)

    subject = deepcopy(participant_profiles[selected_participant])
    subject.setdefault("age_group", "Blank")
    subject.setdefault("gender", "")
    subject["participant_id"] = generate_participant_id(
        subject.get("name", ""), subject.get("age_group", ""), subject.get("gender", "")
    )
    subject.setdefault("has_second_participant", False)
    subject.setdefault("second_name", "")
    subject["second_age_group"] = subject.get("second_age_group", "Blank") or "Blank"
    subject["second_gender"] = subject.get("second_gender", "")
    subject["second_participant_id"] = generate_participant_id(
        subject.get("second_name", ""),
        subject.get("second_age_group", ""),
        subject.get("second_gender", ""),
    )
    subject["second_age_value"] = (
        0 if subject.get("second_age_group") == "Blank" else subject.get("second_age_value", 0)
    )
    subject["experiment_id"] = cfg.get_session_experiment_id()
    subject["profile"] = selected_participant
    experiment = deepcopy(experiment_profiles[selected_experiment])
    experiment["name"] = selected_experiment
    experiment["profile"] = selected_experiment
    environment_profile = deepcopy(
        environment_profiles.get(selected_environment, DEFAULT_ENVIRONMENT_PROFILE)
    )
    environment_profile["profile"] = selected_environment
    actions = filter_blank_actions(action_profiles.get(selected_actions, {}))
    experiment["actions_profile"] = selected_actions
    experiment["actions_list"] = list(actions.keys())
    camera_profile = camera_profiles.get(selected_camera, {})
    depth_camera_profile = depth_camera_profiles.get(selected_depth_camera, {})
    ui_profile = ui_profiles.get(selected_ui, {})

    voice_profile = voice_profiles.get(selected_voice, {})
    wifi_profile = wifi_profiles.get(selected_wifi, [])
    demo_profile = demo_profiles.get(selected_demo, {})
    time_profile = time_profiles.get(selected_time, {})
    time_ref = time_reference.build_time_reference(time_profile)
    time_reference.set_global_time_reference(time_ref)
    if isinstance(wifi_profile, list):
        wifi_profile = {"access_points": wifi_profile}
    elif not isinstance(wifi_profile, dict):
        wifi_profile = {}
    pre_action_csi_duration = float(wifi_profile.get("pre_action_capture_duration", 2.0))
    post_action_csi_duration = float(wifi_profile.get("post_action_capture_duration", 2.0))
    experiment.update(
        {
            "use_webcam": camera_profile.get("use_webcam", False),
            "camera_device": camera_profile.get("camera_device", "0"),
            "use_hand_recognition": camera_profile.get("use_hand_recognition", False),
            "hand_recognition_mode": camera_profile.get("hand_recognition_mode", "live"),
            "hand_model_complexity": camera_profile.get("hand_model_complexity", "light"),
        }
    )
    results_dir = MainWindow.build_results_dir(subject, time_reference=time_ref)
    wifi_capture_info = []
    scenario = str(wifi_profile.get("csi_capture_scenario", "scenario_2")).lower()
    access_points = wifi_profile.get("access_points", []) if wifi_profile else []
    wifi_enabled = bool(access_points) and scenario != "no_collection"
    if wifi_enabled and scenario != "demo":
        expected_duration = _compute_expected_duration(experiment, actions)
        if scenario == "scenario_1":
            capture_duration = expected_duration
        else:
            capture_duration = max(
                1.0,
                pre_action_csi_duration
                + float(experiment.get("action_time", 4))
                + post_action_csi_duration,
            )
        capture_dialog = CSICaptureStartupDialog(
            selected_wifi,
            access_points,
            duration=capture_duration,
            wifi_profile=wifi_profile,
            build_prefix=lambda ap_name: _build_capture_prefix(
                subject, ap_name, time_ref=time_ref
            ),
            log_file_path=results_dir / "csi_captures" / "startup_capture.log",
        )
        capture_dialog.exec_()
        if not getattr(capture_dialog, "successful", False):
            sys.exit(1)
        wifi_capture_info = getattr(capture_dialog, "routers_info", [])

    if scenario == "demo":
        win = DemoWindow(
            subject=subject,
            wifi_profile=wifi_profile,
            demo_profile=demo_profile,
            results_dir=results_dir,
        )
    else:
        win = MainWindow(
            experiment,
            subject,
            actions,
            voice_profile=voice_profile,
            wifi_profile=wifi_profile,
            camera_profile=camera_profile,
            depth_camera_profile=depth_camera_profile,
            ui_profile=ui_profile,
            environment_profile=environment_profile,
            prestarted_wifi=wifi_capture_info if wifi_enabled else [],
            start_wifi_capture=wifi_enabled and not bool(wifi_capture_info),
            results_dir=results_dir,
            time_reference=time_ref,
        )
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
