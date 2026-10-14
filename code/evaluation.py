"""
evaluation.py
===================================
Scores every model output in output/ with the simulator and writes one row per
output item (model x prompt x run x test item) to a CSV file.

Columns:
  tier, model, prompt, run, id   identify the item (run is 1 except for the
                                 three runs of Kimi-K2.7-Code and ChatGPT-5.5)
  permissive        goal reached, all hidden constraints active (path_reach)
  strict            goal reached, every command executed literally
  no_turn_advance   goal reached with one hidden constraint disabled and the
  no_stop_cascade   other three active (one-at-a-time ablation)
  no_revisit_block
  neighbor4         detect() uses the 4-neighbourhood instead of the 8-neighbourhood
  parsed            the output is valid JSON of the form {"commands": [...]}
                    with a non-empty list
  executable        parsed, every command is one of the defined commands and
                    the last one is stop()
  reached_executable  executable and the goal is reached
  no_fallback       goal reached when only the parsed outputs are executed,
                    i.e. without the regular-expression fallback

The permissive, strict and ablation columns use simulator.parse_commands():
<think> blocks and code fences are removed, the JSON is read, and if that fails
the commands are extracted with a regular expression (fallback).

Usage:
    python code/evaluation.py
    python code/evaluation.py --out results/evaluation.csv
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import simulator as sim                  # noqa: E402
from map_generator import generate_map   # noqa: E402

CORPUS = os.path.join(ROOT, "corpus")
OUTPUT = os.path.join(ROOT, "output")

LEGAL = {"start", "move_forward", "move_to_end", "detect",
         "detect_intersection", "turn_right", "turn_left", "stop"}

ABLATION = [
    ("no_turn_advance", {"turn": False}),
    ("no_stop_cascade", {"stop": False}),
    ("no_revisit_block", {"revisit": False}),
    ("neighbor4", {"neighbor8": False}),
]


# ---------------------------------------------------------------------------
# Parsing without the regular-expression fallback
# ---------------------------------------------------------------------------

def strip_wrapper(s):
    """Remove a <think> block and a code fence, as parse_commands() does."""
    t = s
    if "<think>" in t and "</think>" in t:
        t = t.split("</think>")[-1].strip()
    if "```" in t:
        try:
            t = t.split("```")[1]
            if t.startswith("json"):
                t = t[4:]
            t = t.strip()
        except IndexError:
            pass
    return t


def parse_json_only(raw):
    """Commands of an output of the form {"commands": [...]}, else None."""
    s = str(raw).strip()
    if not s:
        return None
    try:
        obj = json.loads(strip_wrapper(s))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    cmds = obj.get("commands")
    if not isinstance(cmds, list) or not cmds:
        return None
    return [str(c).strip() for c in cmds]


def is_executable(cmds):
    """Every command is a defined command and the sequence ends with stop()."""
    for c in cmds:
        m = re.match(r"\s*([a-z_]+)\s*\(", c)
        if not m or m.group(1) not in LEGAL:
            return False
    return cmds[-1].startswith("stop(")


# ---------------------------------------------------------------------------

def reached(cmds, item, mode="permissive", hidden=None):
    """Run the simulator on one command sequence; True if the goal is reached."""
    if not cmds:
        return False
    grid, groups, landmarks, start, goal, pose = item
    sim.set_exec_mode(mode)
    if hidden:
        sim.set_hidden(**hidden)
    try:
        out = sim.simulate(cmds, start, grid, groups, landmarks,
                           goal_cell=goal, start_pose=pose)
        return bool(out["success"])
    except Exception:
        return False
    finally:
        sim.set_exec_mode("permissive")


def load_items():
    """Map, start/goal and fixed start pose of every corpus item, by id."""
    gt = pd.concat([pd.read_csv(p, encoding="utf-8-sig")
                    for p in sorted(glob.glob(os.path.join(CORPUS, "gt_*.csv")))])
    poses = pd.read_csv(os.path.join(CORPUS, "fixed_start_poses.csv"),
                        encoding="utf-8-sig").set_index("id")
    maps, items = {}, {}
    for _, r in gt.iterrows():
        seed = int(r["seed"])
        if seed not in maps:
            maps[seed] = generate_map(seed=seed)
        grid, groups, landmarks, start, goal, _ = maps[seed]
        p = poses.loc[r["id"]]
        pose = ([int(p["cell_row"]), int(p["cell_col"])], p["heading"])
        items[r["id"]] = (grid, groups, landmarks, start, goal, pose)
    return items


def output_files():
    """(tier, model, prompt, run, path) for every file under output/."""
    files = []
    for path in sorted(glob.glob(os.path.join(OUTPUT, "*", "*", "*.csv"))):
        tier = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(re.escape(model) + r"_(v\d)(?:_run(\d))?\.csv",
                         os.path.basename(path))
        if not m:
            raise ValueError("unexpected file name: %s" % path)
        files.append((tier, model, m.group(1), int(m.group(2) or 1), path))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "evaluation.csv"),
                    help="output CSV (default: results/evaluation.csv)")
    args = ap.parse_args()

    items = load_items()
    rows = []
    for tier, model, prompt, run, path in output_files():
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        for _, r in df.iterrows():
            item = items[r["id"]]
            raw = str(r["output"])
            cmds = sim.parse_commands(raw)          # with the regex fallback
            strict_cmds = parse_json_only(raw)      # without it
            row = {"tier": tier, "model": model, "prompt": prompt, "run": run,
                   "id": r["id"],
                   "permissive": reached(cmds, item, "permissive"),
                   "strict": reached(cmds, item, "strict")}
            for name, hidden in ABLATION:
                row[name] = reached(cmds, item, "permissive", hidden)
            row["parsed"] = strict_cmds is not None
            row["executable"] = row["parsed"] and is_executable(strict_cmds)
            row["reached_executable"] = (row["executable"]
                                         and reached(strict_cmds, item, "permissive"))
            row["no_fallback"] = reached(strict_cmds, item, "permissive")
            rows.append(row)
        print("%-6s %-18s %s run%d  permissive %2d/%d" % (
            tier, model, prompt, run,
            sum(x["permissive"] for x in rows[-len(df):]), len(df)), flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("\n%d rows written to %s" % (len(res), args.out))


if __name__ == "__main__":
    main()
