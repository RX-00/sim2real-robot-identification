from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import inspect
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import ProcessedTrajectory
from sysid_mujoco.common import build_actuator_gain_map
from sysid_mujoco.common import build_parameter_dict
from sysid_mujoco.common import get_robot_model_xml_path
from sysid_mujoco.common import get_actuated_joint_and_actuator_names
from sysid_mujoco.common import load_dataset_actuator_gains
from sysid_mujoco.common import load_processed_dataset
from sysid_mujoco.common import processed_to_sysid_trajectory

import config
import mujoco
import mujoco.rollout as rollout
from mujoco import sysid
import numpy as np


PARAMETER_ATTRIBUTES = ("armature", "damping", "frictionloss")
GO2_JOINT_EXPORT_ORDER = (
    "FL_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FR_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "RL_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RR_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
)


@dataclass
class SequenceBundle:
    names: list[str]
    measurements: list[Any]
    controls: list[Any]
    initial_states: list[np.ndarray]


@dataclass
class FitCandidate:
    label: str
    chunk_size: int
    train_sequences: list[Any]
    residual_fn: Any
    initial_params: Any
    opt_params: Any
    opt_result: Any
    validation_rmse: float
    validation_position_rmse_by_traj: dict[str, float]
    validation_position_rmse_by_joint: dict[str, float]


def default_dataset_paths(robot: str) -> list[Path]:
    return sorted((REPO_ROOT / "datasets" / robot).glob("traj_*.pt"))


def default_output_dir(robot: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / robot / f"{timestamp}_v2"


def parse_chunk_size(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"full", "none", "0"}:
        return 0
    chunk_size = int(normalized)
    if chunk_size < 2:
        raise argparse.ArgumentTypeError(
            "Chunk sizes must be `full`/`0` or an integer >= 2."
        )
    return chunk_size


def chunk_label(chunk_size: int) -> str:
    return "full" if chunk_size <= 0 else str(chunk_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Go2 fixed-base MuJoCo joint dynamics over several chunk sizes, "
            "selecting the parameters with the lowest held-out rollout error."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Raw repository datasets (.pt). Defaults to all "
            "`datasets/<robot>/traj_*.pt` files."
        ),
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=2,
        help="Number of sorted trajectories to hold out from fitting.",
    )
    parser.add_argument(
        "--validation-dataset",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Explicit validation datasets. If set, every --dataset path not in "
            "this list is used for fitting."
        ),
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=parse_chunk_size,
        default=[0, 100, 200],
        help="Chunk sizes to compare. Use `full` or `0` for full trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the summary, selected parameters, and generated XML.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("mujoco", "scipy", "scipy_parallel_fd"),
        default="mujoco",
        help="Optimizer backend exposed by mujoco.sysid.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help=(
            "Maximum optimizer iterations. Passed when supported by the "
            "installed mujoco.sysid.optimize signature."
        ),
    )
    parser.add_argument(
        "--damping-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.1, 3.0),
        help="Bounds for each joint damping parameter.",
    )
    parser.add_argument(
        "--armature-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.001, 0.6),
        help="Bounds for each joint armature parameter.",
    )
    parser.add_argument(
        "--frictionloss-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.01, 5.0),
        help="Bounds for each joint frictionloss parameter.",
    )
    return parser.parse_args()


def split_datasets(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    dataset_paths = sorted(args.dataset or default_dataset_paths(args.robot))
    if not dataset_paths:
        raise ValueError(f"No datasets found for robot `{args.robot}`.")

    dataset_paths = [path.resolve() for path in dataset_paths]
    if args.validation_dataset:
        validation_paths = {path.resolve() for path in args.validation_dataset}
        missing = sorted(validation_paths.difference(dataset_paths))
        if missing:
            raise ValueError(
                "Validation datasets must also be present in --dataset: "
                + ", ".join(str(path) for path in missing)
            )
        train_paths = [path for path in dataset_paths if path not in validation_paths]
        return train_paths, sorted(validation_paths)

    if args.validation_count < 1:
        raise ValueError("--validation-count must be at least 1.")
    if args.validation_count >= len(dataset_paths):
        raise ValueError(
            "--validation-count must leave at least one trajectory for fitting."
        )
    return dataset_paths[:-args.validation_count], dataset_paths[-args.validation_count:]


def build_fixed_base_spec_and_model(
    robot: str,
    dataset_for_gains: Path,
    output_dir: Path,
) -> tuple[Any, Any]:
    dataset_kp, dataset_kd = load_dataset_actuator_gains(dataset_for_gains)

    fixed_base_xml = build_fixed_base_model_xml_v2(
        robot,
        output_dir / f"{robot}_fixed_base_sysid_without_gains.xml",
    )
    model_without_gains = mujoco.MjSpec.from_file(str(fixed_base_xml)).compile()
    actuated_joint_names, _ = get_actuated_joint_and_actuator_names(
        mujoco,
        model_without_gains,
    )

    fixed_base_xml = build_fixed_base_model_xml_v2(
        robot,
        output_dir / f"{robot}_fixed_base_sysid.xml",
        actuator_gains=build_actuator_gain_map(
            actuated_joint_names,
            dataset_kp,
            dataset_kd,
        ),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    return fixed_base_spec, fixed_base_spec.compile()


def remove_all_by_tag(root: ET.Element, tag: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)


def absolutize_file_attributes(root: ET.Element, base_dir: Path) -> None:
    for element in root.iter():
        file_attribute = element.get("file")
        if file_attribute is None:
            continue
        file_path = Path(file_attribute)
        if not file_path.is_absolute():
            element.set("file", str((base_dir / file_path).resolve()))


def rewrite_actuators_as_general(
    root: ET.Element,
    actuator_gains: dict[str, tuple[float, float]] | None,
) -> None:
    actuator_element = root.find("actuator")
    source_actuators: list[dict[str, str]] = []
    if actuator_element is not None:
        for actuator in actuator_element:
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue
            actuator_spec = {"joint": joint_name, "gear": "1"}
            actuator_name = actuator.get("name")
            if actuator_name is not None:
                actuator_spec["name"] = actuator_name
            source_actuators.append(actuator_spec)

    remove_all_by_tag(root, "motor")

    joint_ranges = {}
    for joint in root.iter("joint"):
        joint_name = joint.get("name")
        joint_range = joint.get("range")
        if joint_name is not None and joint_range is not None:
            joint_ranges[joint_name] = joint_range

    if actuator_element is None:
        actuator_element = ET.SubElement(root, "actuator")
    else:
        for child in list(actuator_element):
            actuator_element.remove(child)

    for actuator_spec in source_actuators:
        joint_name = actuator_spec["joint"]
        if joint_name in joint_ranges:
            actuator_spec["ctrlrange"] = joint_ranges[joint_name]
        if actuator_gains is not None and joint_name in actuator_gains:
            kp, kd = actuator_gains[joint_name]
            actuator_spec["biastype"] = "affine"
            actuator_spec["gainprm"] = f"{kp:.12g}"
            actuator_spec["biasprm"] = f"0 {-kp:.12g} {-kd:.12g}"
        ET.SubElement(actuator_element, "general", actuator_spec)


def disable_all_collisions(root: ET.Element) -> None:
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def build_fixed_base_model_xml_v2(
    robot: str,
    output_xml: Path,
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> Path:
    source_xml = get_robot_model_xml_path(robot)
    tree = ET.parse(source_xml)
    root = tree.getroot()

    remove_all_by_tag(root, "freejoint")
    remove_all_by_tag(root, "keyframe")
    remove_all_by_tag(root, "accelerometer")
    remove_all_by_tag(root, "gyro")
    remove_all_by_tag(root, "framepos")
    remove_all_by_tag(root, "framequat")
    rewrite_actuators_as_general(root, actuator_gains=actuator_gains)
    absolutize_file_attributes(root, source_xml.parent)
    disable_all_collisions(root)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def chunk_processed_trajectory_v2(
    trajectory: ProcessedTrajectory,
    chunk_size: int,
) -> list[ProcessedTrajectory]:
    if chunk_size <= 0:
        return [trajectory]

    chunks = []
    num_steps = int(trajectory.times.shape[0])
    chunk_index = 0
    for start in range(0, num_steps, chunk_size):
        end = min(start + chunk_size, num_steps)
        if end - start < 2:
            continue
        chunks.append(
            ProcessedTrajectory(
                source_path=trajectory.source_path,
                sequence_name=(
                    f"{trajectory.sequence_name}_chunk_{chunk_index:03d}"
                ),
                times=trajectory.times[start:end] - trajectory.times[start],
                measured_qpos=trajectory.measured_qpos[start:end],
                measured_qvel=trajectory.measured_qvel[start:end],
                desired_qpos=trajectory.desired_qpos[start:end],
                ctrl=trajectory.ctrl[start:end],
                joint_names=list(trajectory.joint_names),
                actuator_names=list(trajectory.actuator_names),
                kp=trajectory.kp,
                kd=trajectory.kd,
            )
        )
        chunk_index += 1
    return chunks or [trajectory]


def build_sequence_bundle(
    model: Any,
    dataset_paths: list[Path],
    chunk_size: int,
) -> SequenceBundle:
    names = []
    measurements = []
    controls = []
    initial_states = []

    for dataset_path in dataset_paths:
        processed = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
        )
        for chunk in chunk_processed_trajectory_v2(processed, chunk_size):
            measurement, control, initial_state = processed_to_sysid_trajectory(
                sysid,
                model,
                chunk,
            )
            names.append(chunk.sequence_name)
            measurements.append(measurement)
            controls.append(control)
            initial_states.append(initial_state)

    return SequenceBundle(
        names=names,
        measurements=measurements,
        controls=controls,
        initial_states=initial_states,
    )


def make_model_sequences(
    robot: str,
    spec: Any,
    bundle: SequenceBundle,
) -> list[Any]:
    return [
        sysid.ModelSequences(
            robot,
            spec,
            bundle.names,
            bundle.initial_states,
            bundle.controls,
            bundle.measurements,
        )
    ]


def optimize_params(
    params: Any,
    residual_fn: Any,
    optimizer: str,
    max_iters: int,
) -> tuple[Any, Any]:
    kwargs = {
        "initial_params": params,
        "residual_fn": residual_fn,
        "optimizer": optimizer,
    }
    try:
        signature = inspect.signature(sysid.optimize)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "max_iters" in signature.parameters:
        kwargs["max_iters"] = max_iters
    return sysid.optimize(**kwargs)


def joint_dynamics_from_params(
    params: Any,
    joint_names: list[str],
) -> dict[str, dict[str, float]]:
    dynamics: dict[str, dict[str, float]] = {
        joint_name: {} for joint_name in sorted(joint_names)
    }
    for joint_name in sorted(joint_names):
        for attribute in PARAMETER_ATTRIBUTES:
            parameter_name = f"{joint_name}_{attribute}"
            dynamics[joint_name][attribute] = float(
                np.asarray(params[parameter_name].value).reshape(-1)[0]
            )
    return dynamics


def apply_joint_dynamics_to_spec(
    spec: Any,
    dynamics: dict[str, dict[str, float]],
) -> None:
    for joint_name, values in dynamics.items():
        joint = spec.joint(joint_name)
        for attribute, value in values.items():
            current = getattr(joint, attribute)
            try:
                current[0] = value
            except (TypeError, IndexError):
                setattr(joint, attribute, value)


def compile_model_with_params(
    base_spec: Any,
    params: Any,
    joint_names: list[str],
) -> Any:
    spec = base_spec.copy()
    apply_joint_dynamics_to_spec(
        spec,
        joint_dynamics_from_params(params, joint_names),
    )
    return spec.compile()


def rollout_joint_positions(
    model: Any,
    processed: ProcessedTrajectory,
) -> np.ndarray:
    data = mujoco.MjData(model)
    initial_state = sysid.create_initial_state(
        model,
        processed.measured_qpos[0],
        processed.measured_qvel[0],
    )
    controls = processed.ctrl[:-1]
    _state, sensor = rollout.rollout(model, data, initial_state, controls)
    sensor = np.squeeze(sensor, axis=0)
    if sensor.ndim == 1:
        sensor = sensor.reshape(-1, 1)
    return sensor


def evaluate_validation_rollouts(
    base_spec: Any,
    opt_params: Any,
    joint_names: list[str],
    validation_paths: list[Path],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    model = compile_model_with_params(base_spec, opt_params, joint_names)

    squared_error_sum = 0.0
    sample_count = 0
    by_traj = {}
    joint_squared_error = np.zeros(len(joint_names), dtype=np.float64)
    joint_sample_count = np.zeros(len(joint_names), dtype=np.float64)
    predictions = {}

    for dataset_path in validation_paths:
        processed = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
        )
        predicted_qpos = rollout_joint_positions(model, processed)
        measured_qpos = processed.measured_qpos
        n_steps = min(predicted_qpos.shape[0], measured_qpos.shape[0])
        n_joints = min(predicted_qpos.shape[1], measured_qpos.shape[1])
        error = predicted_qpos[:n_steps, :n_joints] - measured_qpos[
            :n_steps,
            :n_joints,
        ]

        by_traj[processed.sequence_name] = float(np.sqrt(np.mean(error**2)))
        predictions[processed.sequence_name] = predicted_qpos[:n_steps, :n_joints]
        squared_error_sum += float(np.sum(error**2))
        sample_count += int(error.size)
        joint_squared_error[:n_joints] += np.sum(error**2, axis=0)
        joint_sample_count[:n_joints] += n_steps

    if sample_count == 0:
        raise RuntimeError("Validation rollout produced no comparable samples.")

    by_joint = {
        joint_name: float(
            np.sqrt(joint_squared_error[index] / joint_sample_count[index])
        )
        for index, joint_name in enumerate(joint_names)
        if joint_sample_count[index] > 0
    }
    total_rmse = float(np.sqrt(squared_error_sum / sample_count))
    return total_rmse, by_traj, by_joint, predictions


def save_identified_parameters_py(
    output_path: Path,
    dynamics: dict[str, dict[str, float]],
) -> None:
    ordered_joint_names = [
        joint_name for joint_name in GO2_JOINT_EXPORT_ORDER if joint_name in dynamics
    ]
    ordered_joint_names.extend(
        joint_name for joint_name in sorted(dynamics) if joint_name not in ordered_joint_names
    )
    lines = [
        "GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS: dict[str, dict[str, float]] = {",
    ]
    for joint_name in ordered_joint_names:
        lines.append(f'    "{joint_name}": {{')
        for attribute in PARAMETER_ATTRIBUTES:
            lines.append(
                f'        "{attribute}": {dynamics[joint_name][attribute]:.5f},'
            )
        lines.append("    },")
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(
    output_path: Path,
    candidates: list[FitCandidate],
    best: FitCandidate,
    train_paths: list[Path],
    validation_paths: list[Path],
) -> None:
    lines = [
        "# MuJoCo SysID v2 Summary",
        "",
        f"Selected chunk size: `{best.label}`",
        f"Selected validation RMSE: `{best.validation_rmse:.8f}` rad",
        "",
        "## Fit/Validation Split",
        "",
        "Training trajectories:",
    ]
    lines.extend(f"- `{path}`" for path in train_paths)
    lines.append("")
    lines.append("Validation trajectories:")
    lines.extend(f"- `{path}`" for path in validation_paths)
    lines.extend(
        [
            "",
            "## Candidate Validation RMSE",
            "",
            "| chunk size | validation RMSE (rad) |",
            "| --- | ---: |",
        ]
    )
    for candidate in candidates:
        lines.append(f"| {candidate.label} | {candidate.validation_rmse:.8f} |")
    lines.extend(["", "## Selected Per-Trajectory Validation RMSE", ""])
    for name, rmse in best.validation_position_rmse_by_traj.items():
        lines.append(f"- `{name}`: `{rmse:.8f}` rad")
    lines.extend(["", "## Selected Per-Joint Validation RMSE", ""])
    for name, rmse in best.validation_position_rmse_by_joint.items():
        lines.append(f"- `{name}`: `{rmse:.8f}` rad")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run MuJoCo system identification with validation-based chunk selection.

    Default usage from the repository root:

        python3 sysid_mujoco/my_fit_v2.py

    By default this script loads all `datasets/<robot>/traj_*.pt` files, holds
    out the last two sorted trajectories for validation, fits the remaining
    trajectories with three candidate horizons (`full`, `100`, and `200`
    samples), then selects the fitted parameters with the lowest held-out joint
    position rollout RMSE. The model matches `my_fit.py`: fixed-base robot,
    disabled collisions, MuJoCo affine PD actuators driven by desired joint
    positions, and joint-position-only tracking.

    Useful overrides:

        python3 sysid_mujoco/my_fit_v2.py --validation-count 1
        python3 sysid_mujoco/my_fit_v2.py --chunk-sizes full 100 200
        python3 sysid_mujoco/my_fit_v2.py --validation-dataset datasets/go2/traj_9.pt

    Outputs are written to `sysid_mujoco/results/<robot>/<timestamp>_v2/` unless
    `--output-dir` is provided. The folder contains the selected parameters in
    `identified_joint_dynamics.py`, a validation summary, and the generated
    fixed-base XMLs used for the run. Plotting and animation are intentionally
    handled by `sysid_mujoco/visualize_fit_v2.py` so optimization can complete
    without any rendering dependencies.
    """
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.robot)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, validation_paths = split_datasets(args)
    if not train_paths:
        raise ValueError("No training trajectories remain after validation split.")
    if not validation_paths:
        raise ValueError("At least one validation trajectory is required.")

    fixed_base_spec, fixed_base_model = build_fixed_base_spec_and_model(
        args.robot,
        train_paths[0],
        output_dir,
    )
    joint_names = [fixed_base_model.joint(i).name for i in range(fixed_base_model.njnt)]

    bounds = {
        "damping": tuple(float(value) for value in args.damping_bounds),
        "armature": tuple(float(value) for value in args.armature_bounds),
        "frictionloss": tuple(float(value) for value in args.frictionloss_bounds),
    }

    unique_chunk_sizes = []
    for chunk_size in args.chunk_sizes:
        if chunk_size not in unique_chunk_sizes:
            unique_chunk_sizes.append(chunk_size)

    print("Training trajectories:")
    for path in train_paths:
        print(f"  - {path}")
    print("Validation trajectories:")
    for path in validation_paths:
        print(f"  - {path}")
    print("Chunk sizes:", ", ".join(chunk_label(size) for size in unique_chunk_sizes))

    candidates = []
    for chunk_size in unique_chunk_sizes:
        label = chunk_label(chunk_size)
        print(f"\n=== Fitting chunk size: {label} ===")

        train_bundle = build_sequence_bundle(
            model=fixed_base_model,
            dataset_paths=train_paths,
            chunk_size=chunk_size,
        )
        model_sequences = make_model_sequences(
            args.robot,
            fixed_base_spec,
            train_bundle,
        )
        params = build_parameter_dict(
            sysid=sysid,
            model=fixed_base_model,
            joint_names=joint_names,
            bounds=bounds,
        )
        residual_fn = sysid.build_residual_fn(models_sequences=model_sequences)
        opt_params, opt_result = optimize_params(
            params=params,
            residual_fn=residual_fn,
            optimizer=args.optimizer,
            max_iters=args.max_iters,
        )
        validation_rmse, by_traj, by_joint, _predictions = evaluate_validation_rollouts(
            base_spec=fixed_base_spec,
            opt_params=opt_params,
            joint_names=joint_names,
            validation_paths=validation_paths,
        )
        print(f"Validation RMSE for chunk {label}: {validation_rmse:.8f} rad")

        candidates.append(
            FitCandidate(
                label=label,
                chunk_size=chunk_size,
                train_sequences=model_sequences,
                residual_fn=residual_fn,
                initial_params=params,
                opt_params=opt_params,
                opt_result=opt_result,
                validation_rmse=validation_rmse,
                validation_position_rmse_by_traj=by_traj,
                validation_position_rmse_by_joint=by_joint,
            )
        )

    best = min(candidates, key=lambda candidate: candidate.validation_rmse)
    best_dynamics = joint_dynamics_from_params(best.opt_params, joint_names)

    print(
        f"\nSelected chunk size {best.label} with validation RMSE "
        f"{best.validation_rmse:.8f} rad"
    )

    save_identified_parameters_py(
        output_dir / "identified_joint_dynamics.py",
        best_dynamics,
    )
    save_summary(
        output_dir / "summary.md",
        candidates,
        best,
        train_paths,
        validation_paths,
    )

    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
