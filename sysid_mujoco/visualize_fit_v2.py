from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import load_processed_dataset
from sysid_mujoco.my_fit_v2 import apply_joint_dynamics_to_spec

import config
import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
from mujoco import sysid
import numpy as np

try:
    import mediapy as media
except ImportError:
    media = None


def default_dataset_paths(robot: str) -> list[Path]:
    return sorted((REPO_ROOT / "datasets" / robot).glob("traj_*.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validation plots and before/after MuJoCo rollout videos "
            "from a completed my_fit_v2.py sysID run."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Completed v2 result directory containing "
            "`identified_joint_dynamics.py` and `<robot>_fixed_base_sysid.xml`."
        ),
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
        help=(
            "Number of sorted trajectories to visualize as validation data. "
            "Use the same value used during fitting."
        ),
    )
    parser.add_argument(
        "--validation-dataset",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Explicit validation datasets. If set, every listed trajectory is "
            "visualized."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip measured-vs-simulated joint position PNG plots.",
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Skip MP4 rollout animations.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="FPS for rendered validation videos.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=640,
        help="Rendered validation video width.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=480,
        help="Rendered validation video height.",
    )
    return parser.parse_args()


def validation_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.validation_dataset:
        return sorted(path.resolve() for path in args.validation_dataset)

    dataset_paths = sorted(args.dataset or default_dataset_paths(args.robot))
    if not dataset_paths:
        raise ValueError(f"No datasets found for robot `{args.robot}`.")
    if args.validation_count < 1:
        raise ValueError("--validation-count must be at least 1.")
    if args.validation_count >= len(dataset_paths):
        raise ValueError(
            "--validation-count must be smaller than the number of datasets."
        )
    return [path.resolve() for path in dataset_paths[-args.validation_count:]]


def load_identified_joint_dynamics(
    parameter_path: Path,
) -> dict[str, dict[str, float]]:
    module_ast = ast.parse(parameter_path.read_text(encoding="utf-8"))
    for node in module_ast.body:
        value_node = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS"
                ):
                    value_node = node.value
                    break
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS"
        ):
            value_node = node.value
        if value_node is not None:
            return ast.literal_eval(value_node)
    raise KeyError(
        f"{parameter_path} does not define GO2_SYSID_IDENTIFIED_JOINT_DYNAMICS."
    )


def visual_xml_with_framebuffer(
    source_xml: Path,
    output_xml: Path,
    width: int,
    height: int,
) -> Path:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_element = visual.find("global")
    if global_element is None:
        global_element = ET.SubElement(visual, "global")
    global_element.set("offwidth", str(max(640, width)))
    global_element.set("offheight", str(max(480, height)))
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def compile_model_with_dynamics(
    base_spec: Any,
    dynamics: dict[str, dict[str, float]] | None,
) -> Any:
    spec = base_spec.copy()
    if dynamics is not None:
        apply_joint_dynamics_to_spec(spec, dynamics)
    return spec.compile()


def set_body_rgba(body: Any, rgba: list[float]) -> None:
    for geom in body.geoms:
        geom.rgba = rgba
    for child in body.bodies:
        set_body_rgba(child, rgba)


def compile_colored_model(
    base_spec: Any,
    dynamics: dict[str, dict[str, float]] | None,
    rgba: list[float],
) -> Any:
    spec = base_spec.copy()
    if dynamics is not None:
        apply_joint_dynamics_to_spec(spec, dynamics)
    set_body_rgba(spec.worldbody, rgba)
    return spec.compile()


def rollout_joint_positions(
    model: Any,
    processed: Any,
) -> np.ndarray:
    data = mujoco.MjData(model)
    initial_state = sysid.create_initial_state(
        model,
        processed.measured_qpos[0],
        processed.measured_qvel[0],
    )
    state, sensor = rollout.rollout(model, data, initial_state, processed.ctrl[:-1])
    del state
    sensor = np.squeeze(sensor, axis=0)
    if sensor.ndim == 1:
        sensor = sensor.reshape(-1, 1)
    return sensor


def save_joint_position_plot(
    output_path: Path,
    base_spec: Any,
    dynamics: dict[str, dict[str, float]],
    validation_path: Path,
) -> None:
    opt_model = compile_model_with_dynamics(base_spec, dynamics)
    joint_names = [opt_model.joint(i).name for i in range(opt_model.njnt)]
    processed = load_processed_dataset(
        dataset_path=validation_path,
        model=opt_model,
        mujoco=mujoco,
    )
    predicted_qpos = rollout_joint_positions(opt_model, processed)
    measured_qpos = processed.measured_qpos
    n_steps = min(predicted_qpos.shape[0], measured_qpos.shape[0])
    n_joints = min(predicted_qpos.shape[1], measured_qpos.shape[1])
    times = processed.times[:n_steps]

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    for index, ax in enumerate(axes.ravel()[:n_joints]):
        ax.plot(
            times,
            measured_qpos[:n_steps, index],
            color="#111827",
            linewidth=1.0,
            label="measured",
        )
        ax.plot(
            times,
            predicted_qpos[:n_steps, index],
            color="#2563eb",
            linewidth=0.9,
            label="simulated",
        )
        ax.set_title(joint_names[index], fontsize=9)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Held-out rollout: {validation_path.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_rollout_animation(
    output_path: Path,
    base_spec: Any,
    dynamics: dict[str, dict[str, float]],
    validation_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    if media is None:
        raise RuntimeError("mediapy is required for MP4 animation output.")

    initial_model = compile_colored_model(
        base_spec,
        dynamics=None,
        rgba=[1.0, 0.2, 0.2, 0.62],
    )
    opt_model = compile_colored_model(
        base_spec,
        dynamics=dynamics,
        rgba=[0.2, 0.4, 1.0, 0.70],
    )
    processed = load_processed_dataset(
        dataset_path=validation_path,
        model=opt_model,
        mujoco=mujoco,
    )
    initial_state = sysid.create_initial_state(
        opt_model,
        processed.measured_qpos[0],
        processed.measured_qvel[0],
    )
    models = [initial_model, opt_model]
    datas = [mujoco.MjData(model) for model in models]
    state, _sensor = rollout.rollout(models, datas, initial_state, processed.ctrl[:-1])
    frames = sysid.render_rollout(
        models,
        datas[0],
        state,
        framerate=fps,
        width=width,
        height=height,
    )
    media.write_video(str(output_path), frames, fps=fps)


def main() -> None:
    """Generate plots and animations for a completed `my_fit_v2.py` run.

    Example:

        python3 sysid_mujoco/visualize_fit_v2.py \
            --output-dir sysid_mujoco/results/go2/<timestamp>_v2

    The script does not run optimization. It loads the saved fixed-base XML and
    `identified_joint_dynamics.py`, then regenerates validation joint plots and
    red-vs-blue before/after rollout MP4s for the held-out trajectories.
    """
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_xml = output_dir / f"{args.robot}_fixed_base_sysid.xml"
    parameter_path = output_dir / "identified_joint_dynamics.py"
    if not source_xml.exists():
        raise FileNotFoundError(f"Missing fixed-base XML: {source_xml}")
    if not parameter_path.exists():
        raise FileNotFoundError(f"Missing identified parameters: {parameter_path}")

    visual_xml = visual_xml_with_framebuffer(
        source_xml,
        output_dir / f"{args.robot}_fixed_base_sysid_visual.xml",
        args.render_width,
        args.render_height,
    )
    base_spec = mujoco.MjSpec.from_file(str(visual_xml))
    dynamics = load_identified_joint_dynamics(parameter_path)
    validation_paths = validation_paths_from_args(args)

    if args.skip_animations:
        animation_error = None
    elif media is None:
        animation_error = "mediapy is not installed; skipping MP4 animations."
        print(animation_error)
    else:
        animation_error = None

    for validation_path in validation_paths:
        if not args.skip_plots:
            plot_path = output_dir / f"validation_{validation_path.stem}_joint_positions.png"
            save_joint_position_plot(plot_path, base_spec, dynamics, validation_path)
            print(f"Wrote {plot_path}")

        if not args.skip_animations and animation_error is None:
            video_path = output_dir / f"validation_{validation_path.stem}_initial_vs_opt.mp4"
            save_rollout_animation(
                video_path,
                base_spec,
                dynamics,
                validation_path,
                args.render_fps,
                args.render_width,
                args.render_height,
            )
            print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
