#!/usr/bin/env python3
"""Diagnose the heading-held expansion on ``slots``: is the turning region selected, is rotation tried there, is the
turn cap enough, and why do rotated candidates fail.  Per iteration it logs the selected region, the fixed-heading
translation outcome (step taken), and the rotation retry outcome."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_coupling import route_clearance, scenario
from formation.region_coupling import CouplingConfig, HeadingHoldRRTStar
from formation.types import Pose2D, wrap_to_pi

SLOT_X = 3.0
"""Centre of the horizontal-wall slot (wall y = 0); the vertical-wall slot is at x = -2, y = -2.5."""


def chamber(x, y):
    """Lower chamber after the first slot."""
    return x > -1.6 and y < 0.0


def turning(x, y):
    """Below the horizontal slot, where the column must turn to about +-90 deg."""
    return abs(x - SLOT_X) < 1.5 and -2.0 < y < 0.0


def upright(yaw, tol=math.radians(20.0)):
    return abs(abs(wrap_to_pi(yaw)) - math.pi / 2) < tol


class Logged(HeadingHoldRRTStar):
    def _try_steps(self, cell, c_star, direction, length, yaw):
        new, portal, attempt = super()._try_steps(cell, c_star, direction, length, yaw)
        self._calls.append(dict(yaw=yaw, ok=new is not None, attempt=attempt,
                                advance=min(self.config.step * 0.5 ** (attempt - 1), length) if new is not None else 0.0,
                                valid_seed=None))
        return new, portal, attempt

    def _steer(self, parent, sample):
        self._calls = []
        result = super()._steer(parent, sample)
        cell = self.cells[parent]
        entry = self.entry[parent]
        row = dict(px=cell.pose.x, py=cell.pose.y, pyaw=cell.pose.yaw, eyaw=entry.yaw, sx=sample.x, sy=sample.y,
                   syaw=sample.yaw, covered=not self._calls)
        if self._calls:
            row.update(trans_ok=self._calls[0]["ok"], trans_attempt=self._calls[0]["attempt"],
                       advance=self._calls[0]["advance"], rot_tried=len(self._calls) > 1)
            if len(self._calls) > 1:
                row.update(rot_yaw=self._calls[1]["yaw"], rot_ok=self._calls[1]["ok"])
        self.log.append(row)
        return result


def run(task):
    seed, cap = task
    Logged.max_turn = cap
    map_data, slots, start = scenario("slots")
    planner = Logged(map_data, slots, start, CouplingConfig(seed=seed, checkpoints=(6000,)))
    planner.log = []
    result = planner.plan()
    log = planner.log
    cells = planner.cells
    first_chamber = next((k for k, c in enumerate(cells) if chamber(c.pose.x, c.pose.y)), None)
    first_upright = next((k for k, c in enumerate(cells) if turning(c.pose.x, c.pose.y) and upright(c.pose.yaw)), None)
    first_above = next((k for k, c in enumerate(cells) if c.pose.y > 0.3 and abs(c.pose.x - SLOT_X) < 1.5), None)
    sel_chamber = [r for r in log if chamber(r["px"], r["py"])]
    sel_turn = [r for r in log if turning(r["px"], r["py"])]
    steered = [r for r in sel_turn if not r["covered"]]
    trans_ok = [r for r in steered if r["trans_ok"]]
    rot = [r for r in steered if r["rot_tried"]]
    yaws_turn = [abs(wrap_to_pi(c.pose.yaw)) for c in cells if turning(c.pose.x, c.pose.y)]
    route = result["checkpoints"][-1].get("route") or []
    return dict(seed=seed, cap=cap, success=result["success"],
                t_first=result["first"]["online_time_s"] if result["first"] else None,
                it_first=result["first"]["iteration"] if result["first"] else None,
                L=result["checkpoints"][-1].get("optimised_cost"),
                clearance=route_clearance(map_data, slots, route) if route else None,
                regions=len(cells), first_chamber_cell=first_chamber, first_upright_cell=first_upright,
                first_above_cell=first_above,
                turning_cells=len(yaws_turn), turning_upright=sum(upright(y) for y in yaws_turn),
                turning_yaw_hist=np.histogram(np.degrees(yaws_turn), bins=(0, 15, 30, 45, 60, 75, 90.01))[0].tolist(),
                p_select_chamber=len(sel_chamber) / len(log), p_select_turning=len(sel_turn) / len(log),
                turning_covered=sum(r["covered"] for r in sel_turn), turning_steered=len(steered),
                turning_trans_ok=len(trans_ok), turning_rot_tried=len(rot),
                turning_rot_ok=sum(r["rot_ok"] for r in rot),
                turning_advance=[r["advance"] for r in trans_ok],
                turning_trans_attempt=[r["trans_attempt"] for r in trans_ok],
                turning_rot_from=[abs(wrap_to_pi(r["eyaw"])) for r in rot],
                turning_rot_to=[abs(wrap_to_pi(r["rot_yaw"])) for r in rot],
                turning_rot_okflags=[bool(r["rot_ok"]) for r in rot],
                turning_sample_above=sum(r["sy"] > 0 for r in sel_turn),
                above_samples_selected_in_turning=sum(1 for r in log if r["sy"] > 0.3 and turning(r["px"], r["py"])),
                above_samples=sum(1 for r in log if r["sy"] > 0.3),
                stats=result["stats"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--caps", nargs="+", type=float, default=(math.pi / 4, math.pi / 2, math.pi))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "heading_hold" / "slots_diag.jsonl")
    args = parser.parse_args()
    tasks = [(s, c) for c in args.caps for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("w") as stream:
        for row in pool.map(run, tasks):
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
