"""Remote-call budgets with a dry-run count before any spending, and the global Jev cap.

Jev calls are limited twice (design D3): per stage (`configs/budgets.yaml`) and by
`jev_total`, the cap on Jev inferences for the whole study. Both are checked against
a persistent ledger in the sqlite cache, in which every Jev request is recorded before
it is sent, successful or not, so the cap holds across separate invocations and crashes.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)
JEV = "jev"


class BudgetExceeded(RuntimeError):
    def __init__(self, stage: str, backend: str, required: int, allowed: int, what: str = "budget"):
        self.required, self.allowed = required, allowed
        super().__init__(f"{stage}: {backend} needs {required} remote calls but the {what} allows {allowed} "
                         f"(configs/budgets.yaml); no calls were made")


class DryRunStop(Exception):
    """Raised after a dry-run report; the CLI exits successfully."""


def stage_allowance(stage: str | None, backend: str = JEV) -> int:
    return int((config.load("budgets").get(stage) or {}).get(backend, 0)) if stage else 0


class Ledger:
    """Persistent record of every Jev request ever sent."""

    def __init__(self, path: Path | None = None, cap: int | None = None):
        self.path = Path(path or config.LEDGER)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = int(config.load("budgets")["jev_total"] if cap is None else cap)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS jev_ledger (n INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "ts REAL NOT NULL, stage TEXT NOT NULL)")

    def total(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM jev_ledger").fetchone()[0]

    def count(self, stage: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM jev_ledger WHERE stage = ?", (stage,)).fetchone()[0]

    def check(self, stage: str, n: int, allowed: int) -> None:
        """Refuse `n` more requests if they would exceed the stage allowance or the global cap."""
        if n and self.count(stage) + n > allowed:
            raise BudgetExceeded(stage, JEV, self.count(stage) + n, allowed, "stage budget (all runs so far)")
        if n and self.total() + n > self.cap:
            raise BudgetExceeded(stage, JEV, self.total() + n, self.cap, "global Jev cap `jev_total`")

    def reserve(self, stage: str, allowed: int) -> int:
        """Record one request right before it is sent; atomic check-and-insert. Returns its ledger number."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.check(stage, 1, allowed)
            n = self.db.execute("INSERT INTO jev_ledger (ts, stage) VALUES (?, ?)", (time.time(), stage)).lastrowid
            self.db.execute("COMMIT")
            return n
        except BaseException:
            self.db.execute("ROLLBACK")
            raise


class Budget:
    def __init__(self, stage: str, allowed: dict[str, int] | None = None, dry_run: bool = False,
                 ledger: Ledger | None = None):
        self.stage = stage
        self.allowed = dict(allowed if allowed is not None else config.load("budgets").get(stage) or {})
        self.dry_run = dry_run
        self.used: dict[str, int] = {}
        self._ledger = ledger

    @property
    def ledger(self) -> Ledger:
        if self._ledger is None:
            self._ledger = Ledger()
        return self._ledger

    def preflight(self, required: dict[str, int]) -> None:
        """Check the calls a stage is about to make, before making any of them."""
        for name, n in required.items():
            log.info("%s: %s would make %d remote calls (budget %s, used %d)", self.stage, name, n,
                     self.allowed.get(name, 0), self.used.get(name, 0))
        for name, n in required.items():
            if name == JEV:
                self.ledger.check(self.stage, n, self.allowed.get(JEV, 0))
            elif n and self.used.get(name, 0) + n > self.allowed.get(name, 0):
                raise BudgetExceeded(self.stage, name, self.used.get(name, 0) + n, self.allowed.get(name, 0))
        report = ", ".join(f"{k}={v} calls" for k, v in required.items()) or "no remote calls"
        if JEV in required:
            report += (f"; Jev ledger: {self.ledger.total()} of {self.ledger.cap} used, "
                       f"{self.ledger.count(self.stage)} of {self.allowed.get(JEV, 0)} for this stage")
        print(f"{'[dry-run] ' if self.dry_run else ''}{self.stage}: {report}")
        if self.dry_run:
            raise DryRunStop

    def charge(self, backend, n_inputs: int) -> None:
        """Account for calls right before they are made; refuses to exceed the budget."""
        if not backend.remote:
            return
        n = backend.calls_needed(n_inputs)
        if backend.name == JEV:  # the Jev backend records each request in the ledger itself
            self.ledger.check(self.stage, n, self.allowed.get(JEV, 0))
            self.used[JEV] = self.used.get(JEV, 0) + n
            return
        total = self.used.get(backend.name, 0) + n
        if total > self.allowed.get(backend.name, 0):
            raise BudgetExceeded(self.stage, backend.name, total, self.allowed.get(backend.name, 0))
        self.used[backend.name] = total
