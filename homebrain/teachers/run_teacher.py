from __future__ import annotations

import argparse

from homebrain.teachers.base import TeacherRunConfig
from homebrain.teachers.mock_teacher import MockTeacher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline HomeBrain teacher over a segment log.")
    parser.add_argument("--teacher", required=True, choices=["mock"], help="Teacher implementation to run.")
    parser.add_argument("--log", required=True, help="Input segment log directory.")
    parser.add_argument("--out", required=True, help="Output teacher artifact directory.")
    args = parser.parse_args(argv)

    teacher = MockTeacher()
    summary = teacher.run(TeacherRunConfig(log_dir=args.log, out_dir=args.out))
    print(
        f"wrote {summary.frame_count} mock teacher frames to "
        f"{summary.manifest_path.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
