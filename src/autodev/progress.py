from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Progress:
    count: int = 0

    def reset(self) -> None:
        self.count = 0

    def tick(self, label: str) -> None:
        self.count += 1
        print(f"✓ [{self.count:02d}] {label}", flush=True)

    def error(self, label: str) -> None:
        # purely cosmetic helper (does not raise)
        self.count += 1
        print(f"✗ [{self.count:02d}] {label}", flush=True)
