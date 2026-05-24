from __future__ import annotations

import argparse
from pathlib import Path
import sys

from homebrain.teachers.scene_teacher import SceneTeacherRunConfig
from homebrain.teachers.moge_scene_teacher import (
    MOGE_DEFAULT_MODEL_ID,
    MoGeSceneTeacherUnavailableError,
    create_moge_scene_teacher,
)
from homebrain.teachers.vggt_scene_teacher import (
    VGGT_DEFAULT_MODEL_ID,
    VGGTSceneTeacherUnavailableError,
    create_vggt_scene_teacher,
)

SCENE_TEACHER_NAMES: tuple[str, ...] = ("vggt", "moge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline HomeBrain scene teacher over a route log.")
    parser.add_argument("--teacher", required=True, choices=SCENE_TEACHER_NAMES, help="Scene teacher implementation.")
    parser.add_argument("--backend", default="real", choices=["real", "fake"], help="Backend; tests use fake.")
    parser.add_argument("--log", required=True, help="Input route log directory.")
    parser.add_argument("--out", required=True, help="Output SceneTeacherPack directory.")
    parser.add_argument("--device", default=None, help="Optional real backend device.")
    parser.add_argument("--model-id", default=None, help="Model id recorded in metadata.")
    parser.add_argument("--model-dir", default=None, help="Optional local teacher checkout or adapter path.")
    parser.add_argument("--checkpoint", default=None, help="Optional local teacher checkpoint path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument("--stride", type=int, default=1, help="Optional frame stride.")
    args = parser.parse_args(argv)

    try:
        if args.teacher == "vggt":
            teacher = create_vggt_scene_teacher(
                backend_name=args.backend,
                model_id=args.model_id or VGGT_DEFAULT_MODEL_ID,
                model_dir=args.model_dir,
                checkpoint=args.checkpoint,
                device=args.device,
                max_frames=args.max_frames,
                stride=args.stride,
            )
        elif args.teacher == "moge":
            teacher = create_moge_scene_teacher(
                backend_name=args.backend,
                model_id=args.model_id or MOGE_DEFAULT_MODEL_ID,
                model_dir=args.model_dir,
                checkpoint=args.checkpoint,
                device=args.device,
                max_frames=args.max_frames,
                stride=args.stride,
            )
        else:
            raise ValueError(f"unsupported scene teacher {args.teacher!r}")
        summary = teacher.run(SceneTeacherRunConfig(log_dir=Path(args.log), out_dir=Path(args.out)))
    except (VGGTSceneTeacherUnavailableError, MoGeSceneTeacherUnavailableError) as exc:
        print(f"{args.teacher} scene teacher unavailable: {exc}", file=sys.stderr)
        return 2

    print(
        "wrote "
        f"{summary.frame_count} {summary.teacher_name} scene-teacher frames "
        f"with {summary.backend} backend to {summary.manifest_path.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
