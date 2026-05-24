from __future__ import annotations

from homebrain.policies.trajectory_scorer_net_v0 import train_main


def main(argv: list[str] | None = None) -> int:
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
