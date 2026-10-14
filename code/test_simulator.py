# -*- coding: utf-8 -*-
"""Unit tests for the simulator's transition semantics.

One case per behaviour: each of the four hidden
constraints is exercised once under permissive execution and once under strict
execution, so that the two modes can be compared side by side.

Run with:
    python test_simulator.py
The run also writes test_maps_overview.png, a picture of every scenario.
"""

import os
import sys
import unittest

import matplotlib
matplotlib.use("Agg")            # no window; the scenarios are drawn to a file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # simulator.py and map_generator.py in this folder

import simulator as sim  # noqa: E402

SCENES = []                       # (test name, scenario), drawn by tearDownModule
OVERVIEW_PNG = os.path.join(HERE, "test_maps_overview.png")

ROAD = getattr(sim, "ROAD", 1)
BUILDING = getattr(sim, "BUILDING", 0)
SIZE = 9


def make_map(roads, goal_cell):
    """Build a tiny map.

    roads     : iterable of (r, c) that are road cells
    goal_cell : the building cell that carries landmark "B"
    Returns (grid, groups, landmarks).
    """
    grid = np.full((SIZE, SIZE), BUILDING, dtype=int)
    for r, c in roads:
        grid[r][c] = ROAD
    grid[goal_cell[0]][goal_cell[1]] = BUILDING
    groups = [[tuple(goal_cell)]]
    landmarks = {0: "B"}             # {group index: landmark name}
    return grid, groups, landmarks


def final_dir(out):
    """The simulator reports final_pos but not final heading; read it from the log."""
    return out["log"][-1]["dir"]


class SimulatorTestCase(unittest.TestCase):
    def setUp(self):
        sim.set_exec_mode("permissive")
        self._scene = None

    def tearDown(self):
        sim.set_exec_mode("permissive")
        if self._scene is not None:
            SCENES.append((self._testMethodName, self._scene))

    def run_seq(self, commands, roads, goal_cell, pose):
        grid, groups, landmarks = make_map(roads, goal_cell)
        start = tuple(pose[0])
        out = sim.simulate(commands, start, grid, groups, landmarks,
                           goal_cell=goal_cell, start_pose=pose)
        self._scene = (grid, groups, landmarks, list(commands),
                       tuple(pose[0]), tuple(goal_cell), out)
        return out


class TestTransitions(SimulatorTestCase):
    """One case per behaviour; permissive and strict side by side."""

    # ------------------------------------------------------------------
    # Commands that are not affected by any hidden switch
    # ------------------------------------------------------------------

    def test_start_heading_comes_from_the_fixed_pose(self):
        """The heading in start() is ignored: the fixed start pose decides it."""
        roads = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
        out = self.run_seq(["start('A', 'north')", "move_forward()"],
                           roads, (5, 5), ([3, 2], "east"))
        self.assertEqual(tuple(out["final_pos"]), (3, 3))   # moves east, not north
        self.assertEqual(final_dir(out), "east")

    def test_move_forward_advances_one_cell(self):
        roads = [(3, 1), (3, 2), (3, 3), (3, 4)]
        out = self.run_seq(["start('A', 'east')", "move_forward()"],
                           roads, (5, 5), ([3, 2], "east"))
        self.assertEqual(tuple(out["final_pos"]), (3, 3))

    # ------------------------------------------------------------------
    # turn_left / turn_right
    # ------------------------------------------------------------------
    TURN_ROADS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 5)]
    TURN_POSE = ([3, 3], "east")
    TURN_CMDS = ["start('A', 'east')", "turn_right()"]

    def test_turn_blocked_advances_then_turns_permissive(self):
        """Permissive: the right turn is blocked, so the robot advances two cells east and turns at (3,5)."""
        out = self.run_seq(self.TURN_CMDS, self.TURN_ROADS, (7, 7),
                           self.TURN_POSE)
        self.assertEqual(tuple(out["final_pos"]), (3, 5))
        self.assertEqual(final_dir(out), "south")
        self.assertIn("turn_advance", out["constraints"])

    def test_turn_blocked_rotates_in_place_strict(self):
        """Strict: in the same scene the robot only rotates and does not move."""
        sim.set_exec_mode("strict")
        out = self.run_seq(self.TURN_CMDS, self.TURN_ROADS, (7, 7),
                           self.TURN_POSE)
        self.assertEqual(tuple(out["final_pos"]), (3, 3))
        self.assertEqual(final_dir(out), "south")
        self.assertNotIn("turn_advance", out["constraints"])

    # ------------------------------------------------------------------
    # no-revisit rule
    # ------------------------------------------------------------------
    REV_ROADS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (2, 4), (2, 5)]
    REV_POSE = ([3, 3], "east")
    REV_CMDS = ["start('A', 'east')", "move_forward()",
                "turn_left()", "turn_left()", "move_forward()"]

    def test_revisit_blocked_permissive(self):
        """Permissive: moving back onto a visited cell is blocked."""
        out = self.run_seq(self.REV_CMDS, self.REV_ROADS, (6, 6), self.REV_POSE)
        self.assertIn("revisit_blocked", out["constraints"])
        self.assertEqual(tuple(out["final_pos"]), (3, 4))

    def test_revisit_allowed_strict(self):
        """Strict: the same sequence may revisit the cell."""
        sim.set_exec_mode("strict")
        out = self.run_seq(self.REV_CMDS, self.REV_ROADS, (6, 6), self.REV_POSE)
        self.assertNotIn("revisit_blocked", out["constraints"])
        self.assertEqual(tuple(out["final_pos"]), (3, 3))

    # ------------------------------------------------------------------
    # detect(): neighbourhood radius
    # ------------------------------------------------------------------
    DET_ROADS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
    DET_POSE = ([3, 3], "east")
    DET_GOAL = (4, 5)
    DET_CMDS = ["start('A', 'east')", "detect('B')"]

    def test_detect_eight_neighbourhood_permissive(self):
        """Permissive: B on the diagonal is in the 8-neighbourhood, so the robot stops at (3,4)."""
        out = self.run_seq(self.DET_CMDS, self.DET_ROADS, self.DET_GOAL,
                           self.DET_POSE)
        self.assertEqual(len(sim.detect_neighbors()), 8)
        self.assertEqual(tuple(out["final_pos"]), (3, 4))

    def test_detect_four_neighbourhood_strict(self):
        """Strict: B is not visible from the same cell (4-neighbourhood), so the robot continues to (3,5)."""
        sim.set_exec_mode("strict")
        out = self.run_seq(self.DET_CMDS, self.DET_ROADS, self.DET_GOAL,
                           self.DET_POSE)
        self.assertEqual(len(sim.detect_neighbors()), 4)
        self.assertEqual(tuple(out["final_pos"]), (3, 5))

    # ------------------------------------------------------------------
    # stop(): multi-condition cascade
    # ------------------------------------------------------------------
    STOP_ROADS = [(3, 1), (3, 2), (3, 3), (3, 4)]
    STOP_POSE = ([3, 3], "east")
    STOP_CMDS = ["start('A', 'east')", "detect('B')", "stop('B')"]

    def test_stop_cascade_rejects_permissive(self):
        """Permissive: B is visible on the diagonal but cannot be entered, so the stop cascade rejects it."""
        out = self.run_seq(self.STOP_CMDS, self.STOP_ROADS, (4, 5),
                           self.STOP_POSE)
        self.assertFalse(out["success"])
        self.assertIn("stop_cannot_enter", out["constraints"])

    def test_stop_succeeds_on_detection_strict(self):
        """Strict: detecting the goal building is enough; it need not be enterable."""
        sim.set_exec_mode("strict")
        out = self.run_seq(self.STOP_CMDS, self.STOP_ROADS, (4, 4),
                           self.STOP_POSE)
        self.assertTrue(out["success"])

    # A second building that carries the goal's name must not end the trial:
    # stop() succeeds only at the goal building of the item, in both modes.
    SAME_ROADS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
    SAME_POSE = ([3, 1], "east")
    SAME_CMDS = ["start('A', 'east')", "stop('B')"]

    def run_same_name(self):
        grid = np.full((SIZE, SIZE), BUILDING, dtype=int)
        for r, c in self.SAME_ROADS:
            grid[r][c] = ROAD
        goal, other = (2, 5), (4, 2)            # both buildings are named "B"
        groups = [[goal], [other]]
        landmarks = {0: "B", 1: "B"}
        out = sim.simulate(self.SAME_CMDS, tuple(self.SAME_POSE[0]), grid, groups,
                           landmarks, goal_cell=goal, start_pose=self.SAME_POSE)
        self._scene = (grid, groups, landmarks, list(self.SAME_CMDS),
                       tuple(self.SAME_POSE[0]), goal, out)
        return out

    def test_stop_ignores_same_name_building_permissive(self):
        """Permissive: another building with the same name does not count; the robot continues to the real goal (seen on the diagonal at (3,4), 8-neighbourhood)."""
        out = self.run_same_name()
        self.assertTrue(out["success"])
        self.assertEqual(tuple(out["final_pos"]), (3, 4))

    def test_stop_ignores_same_name_building_strict(self):
        """Strict: likewise only the real goal building counts (the success criterion is not a hidden constraint)."""
        sim.set_exec_mode("strict")
        out = self.run_same_name()
        self.assertTrue(out["success"])
        self.assertEqual(tuple(out["final_pos"]), (3, 5))

    # ------------------------------------------------------------------
    # End to end
    # ------------------------------------------------------------------

    def test_ground_truth_style_route_reaches_goal(self):
        roads = [(3, 1), (3, 2), (3, 3), (3, 4), (4, 4), (4, 5)]
        cmds = ["start('A', 'east')", "move_forward()", "move_forward()",
                "turn_right()", "move_forward()", "stop('B')"]
        out = self.run_seq(cmds, roads, (5, 5), ([3, 1], "east"))
        self.assertTrue(out["success"])


def heading_track(out):
    """Points in the log where the heading changes: [(position, heading), ...]."""
    seq = []
    for e in out["log"]:
        pos, d = tuple(e["pos"]), e["dir"]
        if not seq or seq[-1][1] != d:
            seq.append((pos, d))
    return seq


def draw_headings(ax, out, size):
    """Draw an arrow at every heading change so that turns are visible."""
    for (r, c), d in heading_track(out):
        dr, dc = sim.DIR_VECTOR[d]
        x, y = c + 0.5, size - 1 - r + 0.5
        ax.arrow(x, y, dc * 0.55, -dr * 0.55,
                 head_width=0.40, head_length=0.26, fc="#111111",
                 ec="white", lw=1.6, zorder=8, length_includes_head=True)


def tearDownModule():
    """After all tests, draw every scenario in one overview image for visual checking."""
    if not SCENES:
        return
    n = len(SCENES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2))
    axes = axes.ravel()
    for ax, (name, sc) in zip(axes, SCENES):
        grid, groups, landmarks, cmds, pose, goal, out = sc
        sim.draw_map_with_simulation(grid, groups, landmarks, out,
                                     start=pose, goal=goal, title="", ax=ax)
        draw_headings(ax, out, grid.shape[0])
        track = " -> ".join(d for _, d in heading_track(out))
        text = " ".join(cmds)
        if len(text) > 58:
            text = text[:55] + "..."
        ax.set_title(f"{name}\n{text}\nheading: {track}", fontsize=5.2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0.028, 1, 1])
    fig.text(0.5, 0.010,
             "S = start    G = goal    R = where the robot stopped    "
             "orange line = path taken    black arrow = heading after each change",
             ha="center", fontsize=9)
    fig.savefig(OVERVIEW_PNG, dpi=170)
    plt.close(fig)
    print("\nOverview saved: %s  (%d scenarios)" % (OVERVIEW_PNG, n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
