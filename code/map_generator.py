"""
map_generator.py
─────────────────
Generates a random 20x20 2D outdoor map with:
  - Road network (randomized grid pattern)
  - Building groups (single and merged cells)
  - Landmark labels (all road-facing buildings, LANDMARK_DENSITY = 1.0)
  - Start and Goal buildings (filled green and red in the image)
  - A* shortest path from Start to Goal (computed, not drawn)

The map is fully determined by the seed. The seed of each map is the number
in its name (map_seed<seed>.png), which is also the `seed` column
of corpus/gt_*.csv.

Usage:
    python map_generator.py <seed>
        e.g. python map_generator.py 2
        Prints a summary and saves map_seed<seed>.png (150 dpi) to the
        current directory.

    from map_generator import generate_map
    grid, groups, landmarks, start, goal, path = generate_map(seed=2)
        Returns the map without drawing it (used by the simulator).

Requirements: numpy, matplotlib. Labels use the font "Microsoft YaHei"
(Windows). On other systems the Chinese labels in the image may not render;
the map returned by generate_map() is not affected.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import random
import heapq
import sys

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# ═══════════════════════════════════════════
#  Config (edit here)
# ═══════════════════════════════════════════

GRID_SIZE        = 20     # Grid dimensions (NxN)
ROAD             = 1      # Road cell value
BUILDING         = 0      # Building cell value
MIN_BLOCK        = 2      # Minimum gap between roads
MAX_BLOCK        = 4      # Maximum gap between roads
LANDMARK_DENSITY = 1.0    # 1.0 = all road-facing buildings get a label
MAX_GROUP_SIZE   = 4      # Maximum cells per building group
MIN_DISTANCE     = 12     # Minimum Manhattan distance between start and goal

LANDMARK_NAMES = [
    # Food & Drink
    "餐厅", "咖啡厅", "面包店", "快餐店", "火锅店",
    "寿司店", "拉面店", "茶餐厅", "烧烤店", "甜品店",
    "奶茶店", "比萨店", "汉堡店", "饺子馆",
    "炸鸡店", "便当店", "粥店",  "小吃店",
    # Retail
    "超市", "便利店", "商场", "书店", "文具店",
    "花店", "水果店", "玩具店", "眼镜店", "鞋店",
    "服装店", "珠宝店",  "手机店", "电器店",
    "家具店", "五金店", "宠物店",  "礼品店",
    # Medical
    "医院", "药店", "诊所", "牙科", "眼科",
    "中医馆", 
    # Finance / Public
    "银行", "邮局", "ATM", "警察局", "消防站",
    "市政厅", "法院", "税务局",  "居委会",
    # Education
    "学校", "幼儿园", "图书馆", "培训班", "大学",
    "书法班",  "美术班",
    # Transport
    "公交站", "地铁站", "停车场", "加油站", "充电站",
     "汽车站", "渡口",
    # Leisure / Sports
    "电影院", "公园", "健身房", "游泳馆", "篮球场",
    "网球场", "台球厅", "游乐场", "KTV", "剧院",
    "博物馆", "美术馆", "科技馆", "动物园", "植物园",
    # Accommodation / Services
    "酒店", "民宿", "美容院", "理发店", "洗衣店",
    "照相馆", "打印店", "旅行社", 
    # Religious / Other
    "寺庙", "教堂", "清真寺", "广场", "市场",
    "菜市场", "夜市", 
]

# ═══════════════════════════════════════════
#  Step 1: Initialize grid
# ═══════════════════════════════════════════

def init_grid(size):
    """Create a 2D grid filled with BUILDING (0)."""
    return np.zeros((size, size), dtype=int)

# ═══════════════════════════════════════════
#  Step 2: Place roads at random intervals
# ═══════════════════════════════════════════

def make_road_positions(size, min_block, max_block):
    """
    Generate road row/col positions at random intervals.
    """
    positions = [0]
    current = 0
    while True:
        gap = random.randint(min_block, max_block)
        current += gap
        if current >= size:
            break
        positions.append(current)
    last = positions[-1]
    if (size - 1) - last >= min_block:
        positions.append(size - 1)
    return positions

def place_roads(grid, seed=None):
    """Place horizontal and vertical roads at random intervals."""
    if seed is not None:
        random.seed(seed)
    size = grid.shape[0]
    road_rows = make_road_positions(size, MIN_BLOCK, MAX_BLOCK)
    road_cols = make_road_positions(size, MIN_BLOCK, MAX_BLOCK)
    for r in road_rows:
        for c in range(size):
            grid[r][c] = ROAD
    for c in road_cols:
        for r in range(size):
            grid[r][c] = ROAD
    return grid, road_rows, road_cols

# ═══════════════════════════════════════════
#  Step 3: Randomize roads
# ═══════════════════════════════════════════

def get_road_neighbors(grid, r, c):
    size = grid.shape[0]
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD:
            result.append((nr, nc))
    return result

def is_connected(grid):
    size = grid.shape[0]
    start = None
    for r in range(size):
        for c in range(size):
            if grid[r][c] == ROAD:
                start = (r, c)
                break
        if start:
            break
    if start is None:
        return False
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop(0)
        for nr, nc in get_road_neighbors(grid, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited) == int(np.sum(grid == ROAD))

def randomize_roads(grid, road_rows, road_cols, seed=None):
    if seed is not None:
        random.seed(seed)
    size = grid.shape[0]
    intersection_set = {(r, c) for r in road_rows for c in road_cols}
    candidates = [
        (r, c) for r in range(size) for c in range(size)
        if grid[r][c] == ROAD and (r, c) not in intersection_set
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        grid[r][c] = BUILDING
        if not is_connected(grid):
            grid[r][c] = ROAD
    return grid

# ═══════════════════════════════════════════
#  Step 4a: Group building cells
# ═══════════════════════════════════════════

def group_buildings(grid, seed=None):
    if seed is not None:
        random.seed(seed)
    size = grid.shape[0]
    cell_to_group = {}
    groups = []

    for r in range(size):
        for c in range(size):
            if grid[r][c] == BUILDING:
                group = {(r, c)}
                groups.append(group)
                cell_to_group[(r, c)] = len(groups) - 1

    all_cells = list(cell_to_group.keys())
    random.shuffle(all_cells)

    for r, c in all_cells:
        if (r, c) not in cell_to_group:
            continue
        gid = cell_to_group[(r, c)]
        current_group = groups[gid]
        if len(current_group) >= MAX_GROUP_SIZE:
            continue

        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size:
                if grid[nr][nc] == BUILDING and (nr, nc) in cell_to_group:
                    ngid = cell_to_group[(nr, nc)]
                    if ngid != gid:
                        ng = groups[ngid]
                        if len(current_group) + len(ng) <= MAX_GROUP_SIZE:
                            neighbors.append((nr, nc, ngid))
        if not neighbors:
            continue

        if random.random() < 0.5:
            nr, nc, ngid = random.choice(neighbors)
            for cell in groups[ngid]:
                current_group.add(cell)
                cell_to_group[cell] = gid
            groups[ngid] = set()

    return [g for g in groups if len(g) > 0]

# ═══════════════════════════════════════════
#  Step 4b: Assign landmark names
# ═══════════════════════════════════════════

def faces_road(group, grid):
    size = grid.shape[0]
    for r, c in group:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD:
                return True
    return False

def assign_landmarks(groups, grid, density=LANDMARK_DENSITY, seed=None):
    if seed is not None:
        random.seed(seed)
    road_facing = [i for i, g in enumerate(groups) if faces_road(g, grid)]
    n = max(1, int(len(road_facing) * density))
    selected = road_facing[:n] if density >= 1.0 else \
               random.sample(road_facing, min(n, len(road_facing)))

    pool = LANDMARK_NAMES[:]
    while len(pool) < len(selected):
        pool += LANDMARK_NAMES[:]

    names = random.sample(pool, len(selected))
    return {selected[i]: names[i] for i in range(len(selected))}

def group_center(group):
    rows = [r for r, c in group]
    cols = [c for r, c in group]
    return sum(rows)/len(rows), sum(cols)/len(cols)

# ═══════════════════════════════════════════
#  Step 5a: Place start and goal on BUILDINGS
# ═══════════════════════════════════════════

def place_start_goal(grid, groups, seed=None):
    """
    Randomly place start and goal on BUILDING cells that face a road.
    Ensures Manhattan distance >= MIN_DISTANCE between them.
    """
    if seed is not None:
        random.seed(seed)
    size = grid.shape[0]
    
    # Extract building groups facing a road
    road_facing_groups = [g for g in groups if faces_road(g, grid)]
    
    candidate_cells = []
    for g in road_facing_groups:
        # Choose one cell directly adjacent to a road per group (so A* can enter the road)
        for r, c in g:
            is_adj = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == ROAD:
                    is_adj = True
                    break
            if is_adj:
                candidate_cells.append((r, c))
                break # Add one representative cell per group
    
    random.shuffle(candidate_cells)
    for i in range(len(candidate_cells)):
        for j in range(i + 1, len(candidate_cells)):
            r1, c1 = candidate_cells[i]
            r2, c2 = candidate_cells[j]
            if abs(r1 - r2) + abs(c1 - c2) >= MIN_DISTANCE:
                return candidate_cells[i], candidate_cells[j]
    return candidate_cells[0], candidate_cells[-1]

# ═══════════════════════════════════════════
#  Step 5b: A* pathfinding
# ═══════════════════════════════════════════

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    """
    A* shortest path on the grid.
    Walkable cells: ROAD cells AND the final goal cell.
    """
    size = grid.shape[0]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        if g > g_score.get(current, float('inf')):
            continue
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size:
                # Can traverse if it's a road OR the final goal cell
                if grid[nr][nc] == ROAD or (nr, nc) == goal:
                    neighbor = (nr, nc)
                    new_g = g + 1
                    if new_g < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = new_g
                        came_from[neighbor] = current
                        f = new_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, new_g, neighbor))
    return []

# ═══════════════════════════════════════════
#  Draw helpers
# ═══════════════════════════════════════════

def draw_group_outline(ax, group, size, lw=1.2):
    for r, c in group:
        x, y = c, size - 1 - r
        for dr, dc, x1, y1, x2, y2 in [
            (-1, 0,  x,   y+1, x+1, y+1),
            ( 1, 0,  x,   y,   x+1, y  ),
            ( 0,-1,  x,   y,   x,   y+1),
            ( 0, 1,  x+1, y,   x+1, y+1),
        ]:
            if (r+dr, c+dc) not in group:
                ax.plot([x1, x2], [y1, y2],
                        color="black", lw=lw, solid_capstyle='round')

# ═══════════════════════════════════════════
#  Main draw function
# ═══════════════════════════════════════════

def draw_map(grid, groups, landmarks, path=None,
             start=None, goal=None, title="Map", ax=None):
    size = grid.shape[0]
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # --- Identify the groups containing Start and Goal ---
    start_group = next((g for g in groups if start in g), set())
    goal_group = next((g for g in groups if goal in g), set())

    # --- Fill road and building cells ---
    for r in range(size):
        for c in range(size):
            x, y = c, size - 1 - r
            
            # Change background color to highlight Start/Goal groups
            if (r, c) in start_group:
                color = "#00ce53"  # Light green (Start)
            elif (r, c) in goal_group:
                color = "#d78078"  # Light red (Goal)
            elif grid[r][c] == ROAD:
                color = "#aaaaaa"
            else:
                color = "white"
                
            ax.add_patch(patches.Rectangle(
                (x, y), 1, 1, linewidth=0, facecolor=color
            ))

    # --- Draw outer border for all building groups ---
    for gid, group in enumerate(groups):
        draw_group_outline(ax, group, size)

    # --- Draw landmark name labels ---
    for gid, name in landmarks.items():
        cr, cc = group_center(groups[gid])
        ax.text(
            cc + 0.5, size - 1 - cr + 0.5, name,
            ha="center", va="center",
            fontsize=7.5, fontweight="light", color="black"
        )

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    if show:
        plt.tight_layout()

# ═══════════════════════════════════════════
#  Generate a single map
# ═══════════════════════════════════════════

def generate_map(seed=None):
    grid = init_grid(GRID_SIZE)
    grid, road_rows, road_cols = place_roads(grid, seed=seed)
    grid = randomize_roads(grid, road_rows, road_cols, seed=seed)
    groups    = group_buildings(grid, seed=seed)
    landmarks = assign_landmarks(groups, grid, seed=seed)
    # Pass building group information when placing Start and Goal
    start, goal = place_start_goal(grid, groups, seed=seed)
    path = astar(grid, start, goal)
    return grid, groups, landmarks, start, goal, path

# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    seed = int(sys.argv[1])

    grid, groups, landmarks, start, goal, path = generate_map(seed=seed)

    print(f"seed={seed}  "
          f"landmarks={len(landmarks)}  "
          f"start={start}  goal={goal}  "
          f"path={len(path)} steps")

    draw_map(grid, groups, landmarks,
             path=path, start=start, goal=goal,
             title=f"seed={seed}  path={len(path)} steps")

    plt.savefig(f"map_seed{seed}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: map_seed{seed}.png")