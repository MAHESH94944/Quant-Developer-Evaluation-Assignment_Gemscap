from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZScoreRule:
    threshold: float = 2.0
    direction: str = "abs"  # "abs", ">", "<"

    def triggered(self, z: float) -> bool:
        if self.direction == "abs":
            return abs(z) >= self.threshold
        if self.direction == ">":
            return z >= self.threshold
        if self.direction == "<":
            return z <= -self.threshold
        raise ValueError(f"unsupported direction: {self.direction!r}")
