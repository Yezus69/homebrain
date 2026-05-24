from __future__ import annotations

from homebrain.policies.trajectory_scorer_net_v0 import eval_main


def main(argv: list[str] | None = None) -> int:
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
