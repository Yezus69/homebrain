from __future__ import annotations

import argparse
import sys

from homebrain.teachers.base import TeacherRunConfig
from homebrain.teachers.depth_pro_teacher import DepthProUnavailableError
from homebrain.teachers.registry import TEACHER_NAMES, create_teacher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline HomeBrain teacher over a segment log.")
    parser.add_argument("--teacher", required=True, choices=TEACHER_NAMES, help="Teacher implementation to run.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output teacher artifact directory.")
    parser.add_argument(
        "--backend",
        default="real",
        choices=["real", "fake"],
        help="Depth Pro backend. Use 'fake' only for tests; real never downloads weights.",
    )
    parser.add_argument("--device", default=None, help="Optional Depth Pro device, e.g. cpu or cuda.")
    args = parser.parse_args(argv)

    try:
        teacher = create_teacher(args.teacher, backend_name=args.backend, device=args.device)
        summary = teacher.run(TeacherRunConfig(log_dir=args.log, out_dir=args.out))
    except DepthProUnavailableError as exc:
        print(f"Depth Pro teacher unavailable: {exc}", file=sys.stderr)
        return 2

    backend_note = f" with {args.backend} backend" if args.teacher == "depth_pro" else ""
    print(f"wrote {summary.frame_count} {args.teacher} teacher frames{backend_note} to {summary.manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
