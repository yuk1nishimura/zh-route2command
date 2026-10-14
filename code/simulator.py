"""
simulator.py
===================================
Grid simulator that executes a command sequence (start / move_forward /
turn_left / turn_right / detect / detect_intersection / stop) on a map from
map_generator.py and decides whether the robot reaches the goal building.

  - detect() checks the current cell only when it is an intersection;
    otherwise the robot keeps advancing before checking.
  - detect_intersection() checks the current cell before advancing.
  - stop() succeeds only at the goal building of the item.

Two execution modes (set_exec_mode):
  "permissive"  all four hidden constraints are active (turn advance,
                stop cascade, no revisit, 8-neighbourhood); main results.
  "strict"      only what the prompt states is executed.
Each hidden constraint can also be switched off on its own (set_hidden).

Usage:
    from simulator import parse_commands, simulate, set_exec_mode
    result = simulate(commands, start, grid, groups, landmarks,
                      goal_cell=goal, start_pose=(cell, heading))
    result["success"]  -> True / False

    python simulator.py
        Runs a short demonstration on map seed 42.
"""

import sys, os, json

# Plotting is optional: simulate() does not need matplotlib, so the module can be
# imported and the evaluation replayed in a headless environment without it.
try:
    import matplotlib
    matplotlib.use('Agg')

    def _set_cjk_font():
        from matplotlib import font_manager
        candidates = [
            'Noto Sans CJK JP', 'Noto Sans CJK SC',
            'Microsoft YaHei', 'SimHei',
            'PingFang SC', 'Hiragino Sans GB',
            'WenQuanYi Micro Hei', 'DejaVu Sans',
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for font in candidates:
            if font in available:
                matplotlib.rcParams['font.family'] = font
                return
        try:
            import subprocess
            subprocess.run(['apt-get', 'install', '-y', 'fonts-noto-cjk'], capture_output=True)
            font_manager._load_fontmanager(try_read_cache=False)
            matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        except Exception:
            pass

    _set_cjk_font()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    matplotlib = None
    plt = None
    patches = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from map_generator import generate_map, ROAD

CLOCKWISE  = ["north", "east", "south", "west"]
DIR_VECTOR = {
    "north": (-1,  0), "south": ( 1,  0),
    "east":  ( 0,  1), "west":  ( 0, -1),
}

def turn_dir(current, side):
    i = CLOCKWISE.index(current)
    return CLOCKWISE[(i + 1) % 4] if side == "right" else CLOCKWISE[(i - 1) % 4]

def can_move(pos, direction, grid):
    size = grid.shape[0]
    dr, dc = DIR_VECTOR[direction]
    nr, nc = pos[0] + dr, pos[1] + dc
    return 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD

def advance(pos, direction):
    dr, dc = DIR_VECTOR[direction]
    pos[0] += dr; pos[1] += dc

def build_cell_to_landmark(groups, landmarks):
    result = {}
    for gid, name in landmarks.items():
        for cell in groups[gid]:
            result[cell] = name
    return result

def get_adjacent_landmarks(cell, grid, c2l, neighbors=None):
    size = grid.shape[0]
    r, c = cell
    found = []
    for dr, dc in (NEIGHBORS_8 if neighbors is None else neighbors):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            name = c2l.get((nr, nc))
            if name and name not in found:
                found.append(name)
    return found

def get_cardinal_landmarks(cell, grid, c2l):
    size = grid.shape[0]
    r, c = cell
    found = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            name = c2l.get((nr, nc))
            if name and name not in found:
                found.append(name)
    return found

def is_intersection(pos, grid):
    size = grid.shape[0]
    r, c = pos[0], pos[1]
    if grid[r][c] != ROAD: return False
    return sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
               if 0 <= r+dr < size and 0 <= c+dc < size
               and grid[r+dr][c+dc] == ROAD) >= 3

def parse_commands(llm_output):
    import re
    if isinstance(llm_output, list): return llm_output
    if isinstance(llm_output, dict): return llm_output.get("commands", [])
    if isinstance(llm_output, str):
        s = llm_output
        if "<think>" in s and "</think>" in s: s = s.split("</think>")[-1].strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"): s = s[4:]
            s = s.strip()
        try: return json.loads(s).get("commands", [])
        except (json.JSONDecodeError, AttributeError): pass
        return re.findall(r'[a-z_]+\([^)]*\)', s)
    return []

def parse_start(cmd):
    inner = cmd[6:-1]
    parts = [p.strip().strip("\"'") for p in inner.split(",")]
    return (parts[0], parts[1]) if len(parts) >= 2 else (parts[0], "north")

def parse_detect(cmd):
    if cmd == "detect_intersection()": return None
    return cmd[7:-1].strip().strip("\"'")

def parse_stop(cmd):
    return cmd[5:-1].strip().strip("\"'")

MAX_STEPS = 500

# ---------------------------------------------------------------------------
# Execution semantics.
#
# "permissive" applies the four hidden behaviours below.
# "strict" executes only what the prompt states: a turn rotates without
# translating, stop() succeeds as soon as the goal landmark is detected, cells
# may be revisited, and "adjacent" means the four orthogonal neighbours.
# ---------------------------------------------------------------------------
EXEC_MODE = "permissive"          # "permissive" | "strict"

# Four behaviours that the prompt never states. Each can be switched off on its
# own so that its individual contribution can be measured (one-at-a-time ablation).
HIDDEN = {"turn": True, "stop": True, "revisit": True, "neighbor8": True}

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def set_exec_mode(mode):
    global EXEC_MODE, HIDDEN
    assert mode in ("permissive", "strict"), mode
    EXEC_MODE = mode
    on = mode == "permissive"
    HIDDEN = {"turn": on, "stop": on, "revisit": on, "neighbor8": on}
    return EXEC_MODE


def set_hidden(turn=None, stop=None, revisit=None, neighbor8=None):
    """Switch individual hidden behaviours off (one-at-a-time ablation)."""
    global HIDDEN
    for key, val in (("turn", turn), ("stop", stop),
                     ("revisit", revisit), ("neighbor8", neighbor8)):
        if val is not None:
            HIDDEN[key] = bool(val)
    return dict(HIDDEN)


def hidden_state():
    return dict(HIDDEN)


def detect_neighbors():
    """Neighbourhood used by detect(): strict reads 'adjacent' as orthogonal."""
    return NEIGHBORS_8 if HIDDEN["neighbor8"] else NEIGHBORS_4


def choose_start_pose(commands, start_cell, grid, groups, landmarks):
    """Pick the start cell and heading for one item.

    The start pose is a property of the item, so it must be resolved once --
    from the ground-truth sequence -- and then reused for every model.
    """
    c2l = build_cell_to_landmark(groups, landmarks)
    size = grid.shape[0]

    lm_start, direction = parse_start(commands[0])

    first_target = None
    for cmd in commands[1:]:
        if cmd.startswith("detect("):
            t = parse_detect(cmd)
            if t:
                first_target = t
            break

    start_group_cells = [tuple(start_cell)]
    for gid, group in enumerate(groups):
        if tuple(start_cell) in group:
            start_group_cells = list(group)
            break

    pos = None
    dr0, dc0 = DIR_VECTOR[direction]
    candidates = []
    seen_positions = set()
    for sc in start_group_cells:
        for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sc[0] + ddr, sc[1] + ddc
            if (nr, nc) in seen_positions:
                continue
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD:
                nnr, nnc = nr + dr0, nc + dc0
                if 0 <= nnr < size and 0 <= nnc < size and grid[nnr][nnc] == ROAD:
                    seen_positions.add((nr, nc))
                    tr, tc = nr + dr0, nc + dc0
                    score = 0
                    can_reach = False
                    while 0 <= tr < size and 0 <= tc < size and grid[tr][tc] == ROAD:
                        score += 1
                        if first_target and first_target in get_adjacent_landmarks(
                                [tr, tc], grid, c2l, NEIGHBORS_8):
                            can_reach = True
                        tr += dr0
                        tc += dc0
                    candidates.append(([nr, nc], score, can_reach))

    if candidates:
        reachable = [(p, s) for p, s, r in candidates if r]
        if reachable:
            pos = max(reachable, key=lambda x: x[1])[0]
        else:
            pos = max(candidates, key=lambda x: x[1])[0]

    if pos is None:
        for sc in start_group_cells:
            for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = sc[0] + ddr, sc[1] + ddc
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD:
                    pos = [nr, nc]
                    break
            if pos is not None:
                break
    if pos is None:
        pos = list(start_cell)
    return pos, direction


def simulate(commands, start_cell, grid, groups, landmarks, goal_cell=None,
             start_pose=None):
    c2l = build_cell_to_landmark(groups, landmarks)
    size = grid.shape[0]

    # Find which group contains the goal cell (for anti-false-positive)
    goal_group_id = None
    if goal_cell is not None:
        for gid, group in enumerate(groups):
            if tuple(goal_cell) in group:
                goal_group_id = gid
                break

    if not commands or not commands[0].startswith("start("):
        return {"path": [tuple(start_cell)], "log": [], "success": False,
                "final_pos": tuple(start_cell), "steps": 0}

    lm_start, direction = parse_start(commands[0])

    if start_pose is not None:
        # The pose was resolved once per item, before inference, from the map and
        # the ground-truth annotation. Same pose for the ground truth and every
        # model.
        pos = list(start_pose[0])
        direction = start_pose[1]
    else:
        pos, direction = choose_start_pose(commands, start_cell, grid, groups, landmarks)

    path = [tuple(start_cell), tuple(pos)]
    global_visited = {tuple(start_cell), tuple(pos)}
    log  = [{"step": 0, "pos": tuple(pos), "dir": direction,
             "msg": f'start("{lm_start}", "{direction}") -> {tuple(pos)}'}]
    step = 0

    def logit(msg):
        nonlocal step
        step += 1
        log.append({"step": step, "pos": tuple(pos), "dir": direction, "msg": msg})

    constraints = set()   # hidden constraints that bit at least once in this trial

    def _check_backtrack(dr, dc):
        """True if the next cell was already visited.

        Strict execution: the prompt never mentions a no-revisit rule, so the
        rule does not apply.
        """
        if not HIDDEN["revisit"]:
            return False
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < size and 0 <= nc < size and (nr, nc) in global_visited:
            constraints.add("revisit_blocked")
            return True
        return False

    # Track consecutive zero-move detects (false-positive guard)
    _consecutive_zero_detect = 0
    _max_consecutive_zero_detect = 0
    _detect_start_pos = None

    for ci, cmd in enumerate(commands):
        if ci == 0: continue  # skip start command
        next_cmd = commands[ci + 1] if ci + 1 < len(commands) else ""

        # ---- move_forward() ----
        if cmd == "move_forward()":
            _consecutive_zero_detect = 0
            _just_detected = False
            if can_move(pos, direction, grid):
                dr, dc = DIR_VECTOR[direction]
                if _check_backtrack(dr, dc):
                    logit(f"move_forward -> backtrack blocked at {tuple(pos)}")
                else:
                    advance(pos, direction)
                    global_visited.add(tuple(pos))
                    path.append(tuple(pos))
                    logit(f"move_forward -> {tuple(pos)}")
            else:
                logit(f"move_forward -> blocked at {tuple(pos)}")

        # ---- detect("lm") ----
        elif cmd.startswith("detect("):
            _detect_start_pos = tuple(pos)
            lm = parse_detect(cmd)
            logit(f'detect("{lm}") start')

            def _turn_would_fail():
                """Check if next command is a turn that can't execute from current pos"""
                if not HIDDEN["turn"]:
                    # A turn only rotates, so it is always executable.
                    return False
                if next_cmd not in ("turn_left()", "turn_right()"):
                    return False
                td = turn_dir(direction, "left" if next_cmd == "turn_left()" else "right")
                if not can_move(pos, td, grid):
                    constraints.add("turn_guard")
                    return True
                return False

            # Check current cell for landmark (intersection + backtrack check)
            adj = get_adjacent_landmarks(pos, grid, c2l, detect_neighbors())
            if lm in adj:
                dr, dc = DIR_VECTOR[direction]
                # Accept current-cell detection if at intersection OR if next step would backtrack
                if is_intersection(pos, grid) or _check_backtrack(dr, dc):
                    if _turn_would_fail():
                        logit(f'  -> detected "{lm}" at {tuple(pos)} but turn would fail, continuing')
                    else:
                        reason = "intersection" if is_intersection(pos, grid) else "pre-backtrack"
                        logit(f'  -> detected "{lm}" at {tuple(pos)} ({reason})')
                        _consecutive_zero_detect = _consecutive_zero_detect + 1 if tuple(pos) == _detect_start_pos else 0
                        _max_consecutive_zero_detect = max(_max_consecutive_zero_detect, _consecutive_zero_detect)
                        _just_detected = True
                        continue
            itr = 0
            visited = {tuple(pos)}
            while itr < MAX_STEPS:
                itr += 1
                if not can_move(pos, direction, grid):
                    logit(f'  -> blocked at {tuple(pos)}')
                    break
                dr, dc = DIR_VECTOR[direction]
                if _check_backtrack(dr, dc):
                    logit(f'  -> backtrack blocked at {tuple(pos)}')
                    break
                advance(pos, direction)
                t = tuple(pos)
                if t in visited:
                    logit(f'  -> backward at {t}, blocked')
                    break
                visited.add(t)
                global_visited.add(t)
                path.append(t)
                adj = get_adjacent_landmarks(pos, grid, c2l, detect_neighbors())
                if lm in adj:
                    if _turn_would_fail():
                        logit(f'  -> detected "{lm}" at {t} but turn would fail, continuing')
                        # Continue forward to find a position where turn works
                    else:
                        logit(f'  -> detected "{lm}" at {t}')
                        _consecutive_zero_detect = _consecutive_zero_detect + 1 if t == _detect_start_pos else 0
                        _max_consecutive_zero_detect = max(_max_consecutive_zero_detect, _consecutive_zero_detect)
                        _just_detected = True
                        break

        # ---- detect_intersection() ----
        elif cmd == "detect_intersection()":
            _detect_start_pos = tuple(pos)
            _just_detected = False
            logit("detect_intersection() start")
            if is_intersection(pos, grid):
                _consecutive_zero_detect = _consecutive_zero_detect + 1 if tuple(pos) == _detect_start_pos else 0
                _max_consecutive_zero_detect = max(_max_consecutive_zero_detect, _consecutive_zero_detect)
                logit(f'  -> intersection at {tuple(pos)} (current)')
                continue
            itr = 0
            visited = {tuple(pos)}
            while itr < MAX_STEPS:
                itr += 1
                if not can_move(pos, direction, grid):
                    logit(f'  -> blocked at {tuple(pos)}')
                    break
                dr, dc = DIR_VECTOR[direction]
                if _check_backtrack(dr, dc):
                    logit(f'  -> backtrack blocked at {tuple(pos)}')
                    break
                advance(pos, direction)
                t = tuple(pos)
                if t in visited:
                    logit(f'  -> backward at {t}, blocked')
                    break
                visited.add(t)
                global_visited.add(t)
                path.append(t)
                if is_intersection(pos, grid):
                    _consecutive_zero_detect = _consecutive_zero_detect + 1 if t == _detect_start_pos else 0
                    _max_consecutive_zero_detect = max(_max_consecutive_zero_detect, _consecutive_zero_detect)
                    logit(f'  -> intersection at {t}')
                    break

        # ---- move_to_end() ----
        elif cmd == "move_to_end()":
            _consecutive_zero_detect = 0
            _just_detected = False
            logit("move_to_end() start")
            itr = 0
            visited = {tuple(pos)}
            while itr < MAX_STEPS:
                itr += 1
                if not can_move(pos, direction, grid):
                    logit(f"  -> blocked at {tuple(pos)}")
                    break
                dr, dc = DIR_VECTOR[direction]
                if _check_backtrack(dr, dc):
                    logit(f"  -> backtrack blocked at {tuple(pos)}")
                    break
                advance(pos, direction)
                t = tuple(pos)
                if t in visited:
                    logit(f'  -> backward at {t}, blocked')
                    break
                visited.add(t)
                global_visited.add(t)
                path.append(t)

        # ---- turn_right() / turn_left() ----
        elif cmd in ("turn_right()", "turn_left()"):
            side = "right" if cmd == "turn_right()" else "left"
            tag = "turn_right" if side == "right" else "turn_left"
            _consecutive_zero_detect = 0
            _just_detected = False
            new_dir = turn_dir(direction, side)
            if not HIDDEN["turn"]:
                # The prompt says a turn is rotation only: rotate the heading and
                # leave the cell, without inspecting whether the turn is blocked.
                direction = new_dir
                logit(f"{tag} -> facing {direction}")
            elif can_move(pos, new_dir, grid):
                direction = new_dir
                logit(f"{tag} -> facing {direction}")
            else:
                constraints.add("turn_advance")
                itr = 0
                while itr < MAX_STEPS:
                    itr += 1
                    if can_move(pos, new_dir, grid):
                        direction = new_dir
                        logit(f"{tag} -> facing {direction} (after {itr} steps)")
                        break
                    if not can_move(pos, direction, grid):
                        constraints.add("turn_skipped")
                        logit(f"{tag}: blocked, skipping")
                        break
                    advance(pos, direction)
                    path.append(tuple(pos))

        # ---- stop("lm") ----
        elif cmd.startswith("stop("):
            goal_lm = parse_stop(cmd)
            logit(f'stop("{goal_lm}") start')
            itr = 0
            visited = {tuple(pos)}
            while itr < MAX_STEPS:
                itr += 1
                # Check adjacent cells for goal landmark
                r, c = pos[0], pos[1]
                found_goal = False
                for dr, dc in detect_neighbors():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < size and 0 <= nc < size): continue
                    if c2l.get((nr, nc)) == goal_lm:
                        # The detected building must be the goal of the item in
                        # both modes: a building elsewhere that carries the same
                        # name, or a wrong goal name, does not count as reaching
                        # the goal. (This is the success criterion, not one of the
                        # hidden constraints, so strict execution keeps it.)
                        if goal_group_id is None or (nr, nc) in groups[goal_group_id]:
                            found_goal = True
                            break
                if found_goal and not HIDDEN["stop"]:
                    # The prompt states stop() moves forward until the goal is
                    # detected and then stops; nothing else is required.
                    logit(f'  -> "{goal_lm}" detected at {tuple(pos)} [OK, strict]')
                    return {"path": path, "log": log, "success": True,
                            "final_pos": tuple(pos), "steps": step,
                            "constraints": sorted(constraints)}
                if found_goal:
                    # The robot must be able to step into the goal building (BFS up to 2 steps)
                    can_step_in = False
                    visited_bfs = {tuple(pos)}
                    for dr4, dc4 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr4, nc4 = pos[0] + dr4, pos[1] + dc4
                        if not (0 <= nr4 < size and 0 <= nc4 < size): continue
                        # Step 1: directly reach goal
                        if (nr4, nc4) in groups[goal_group_id]:
                            can_step_in = True; break
                        # Step 2: step onto road, then check if goal adjacent
                        if grid[nr4][nc4] == ROAD and (nr4, nc4) not in visited_bfs:
                            visited_bfs.add((nr4, nc4))
                            for dr4b, dc4b in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr4b, nc4b = nr4 + dr4b, nc4 + dc4b
                                if 0 <= nr4b < size and 0 <= nc4b < size and (nr4b, nc4b) in groups[goal_group_id]:
                                    can_step_in = True; break
                        if can_step_in: break
                    if not can_step_in:
                        constraints.add("stop_cannot_enter")
                        logit(f'  -> "{goal_lm}" detected but cannot step in from {tuple(pos)}, keep moving')
                        found_goal = False
                    if found_goal:
                        # Verify goal building has road-accessible side (not sealed)
                        accessible = False
                        for gcell in groups[goal_group_id]:
                            gr, gc = gcell
                            for dr4, dc4 in [(-1,0),(1,0),(0,-1),(0,1)]:
                                ar, ac = gr + dr4, gc + dc4
                                if 0 <= ar < size and 0 <= ac < size and grid[ar][ac] == ROAD:
                                    accessible = True; break
                            if accessible: break
                    if found_goal and accessible:
                        if _max_consecutive_zero_detect >= 3:
                            constraints.add("stop_suspicious_path")
                            logit(f'  -> "{goal_lm}" reached but path suspicious (peak {_max_consecutive_zero_detect} zero-move detects) [REJECTED]')
                            return {"path": path, "log": log, "success": False,
                                    "final_pos": tuple(pos), "steps": step,
                                    "constraints": sorted(constraints),
                                    "failure_reason": "stop_suspicious_path",
                                    "error": f"suspicious_path: peak {_max_consecutive_zero_detect} consecutive zero-move detects"}
                        logit(f'  -> "{goal_lm}" found at {tuple(pos)} [OK]')
                        return {"path": path, "log": log, "success": True,
                                "final_pos": tuple(pos), "steps": step,
                                "constraints": sorted(constraints)}
                    else:
                        constraints.add("stop_sealed")
                        logit(f'  -> "{goal_lm}" at {tuple(pos)} but building sealed, keep moving')
                        found_goal = False
                if not can_move(pos, direction, grid):
                    logit(f'  -> blocked, "{goal_lm}" not found')
                    break
                dr, dc = DIR_VECTOR[direction]
                if _check_backtrack(dr, dc):
                    logit(f'  -> backtrack blocked at {tuple(pos)}')
                    break
                advance(pos, direction)
                t = tuple(pos)
                if t in visited:
                    logit(f'  -> backward at {t}, blocked')
                    break
                visited.add(t)
                global_visited.add(t)
                path.append(t)

        else:
            logit(f"unknown command: {cmd}")

    return {"path": path, "log": log, "success": False,
            "final_pos": tuple(pos), "steps": step,
            "constraints": sorted(constraints),
            "failure_reason": "goal_not_reached"}

# ---- Drawing ----

def _outline_group(ax, group, size, lw=1.2):
    for r, c in group:
        x, y = c, size - 1 - r
        for dr, dc, x1, y1, x2, y2 in [
            (-1,0, x, y+1, x+1, y+1), (1,0, x, y, x+1, y),
            (0,-1, x, y, x, y+1),     (0,1, x+1, y, x+1, y+1),
        ]:
            if (r+dr, c+dc) not in group:
                ax.plot([x1,x2],[y1,y2], color="black", lw=lw, solid_capstyle='round')

def _group_center(group):
    rows = [r for r, c in group]; cols = [c for r, c in group]
    return sum(rows)/len(rows), sum(cols)/len(cols)

def draw_map_with_simulation(grid, groups, landmarks, sim_result,
                              start=None, goal=None, title="Simulation", ax=None):
    size = grid.shape[0]
    show = ax is None
    if ax is None: _, ax = plt.subplots(figsize=(9, 9))
    for r in range(size):
        for c in range(size):
            color = "#aaaaaa" if grid[r][c] == ROAD else "white"
            ax.add_patch(patches.Rectangle((c, size-1-r), 1, 1, linewidth=0, facecolor=color))
    for gid, group in enumerate(groups): _outline_group(ax, group, size)
    for gid, name in landmarks.items():
        cr, cc = _group_center(groups[gid])
        ax.text(cc+0.5, size-1-cr+0.5, name, ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="black")
    sp = sim_result["path"]
    if len(sp) > 1:
        xs = [c+0.5 for r, c in sp]; ys = [size-1-r+0.5 for r, c in sp]
        ax.plot(xs, ys, color="#e67e22", lw=2.5, solid_capstyle='round',
                solid_joinstyle='round', zorder=4)
    if start:
        sr, sc = start
        ax.add_patch(patches.Circle((sc+0.5, size-1-sr+0.5), 0.38,
            facecolor="#2ecc71", edgecolor="white", lw=1.5, zorder=5))
        ax.text(sc+0.5, size-1-sr+0.5, "S", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)
    if goal:
        gr, gc = goal
        ax.add_patch(patches.Circle((gc+0.5, size-1-gr+0.5), 0.38,
            facecolor="#e74c3c", edgecolor="white", lw=1.5, zorder=5))
        ax.text(gc+0.5, size-1-gr+0.5, "G", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)
    fp = sim_result["final_pos"]
    if goal and fp != tuple(goal):
        ax.add_patch(patches.Circle((fp[1]+0.5, size-1-fp[0]+0.5), 0.3,
            facecolor="#3498db", edgecolor="white", lw=1.5, zorder=5))
        ax.text(fp[1]+0.5, size-1-fp[0]+0.5, "R", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
    status = "[OK] reached goal" if sim_result["success"] else "[X] did not reach goal"
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}\n{status}  (steps: {sim_result['steps']})", fontsize=9)
    if show: plt.tight_layout(); plt.show()

# ---- Main ----
if __name__ == "__main__":
    seed = 42
    grid, groups, landmarks, start, goal, _ = generate_map(seed=seed)
    llm_output = {"commands": [
        'start("ATM", "west")', 'move_forward()', 'detect("饺子馆")', 'turn_left()',
        'move_forward()', 'detect("咖啡厅")', 'turn_right()',
        'move_forward()', 'detect("居委会")', 'turn_left()',
        'move_forward()', 'detect("宠物店")', 'turn_left()',
        'move_forward()', 'detect("诊所")', 'turn_right()',
        'move_forward()', 'detect("鞋店")', 'turn_left()',
        'stop("中医馆")',
    ]}
    cmds = parse_commands(llm_output)
    result = simulate(cmds, start, grid, groups, landmarks, goal_cell=goal)
    print(f"seed: {seed}  start: {start}  goal: {goal}")
    print(f"success: {result['success']}  final_pos: {result['final_pos']}")
    for e in result["log"]:
        print(f"  step{e['step']:3d}  {str(e['pos']):<12}  {e['dir']:5s}  {e['msg']}")
