
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import random

def generate_random_dag(num_nodes, max_out_degree=2, seed=None):
    """
    Generates a random Directed Acyclic Graph (DAG).

    :param num_nodes: Number of nodes in the graph.
    :param max_out_degree: Maximum number of edges going out from any node.
    :param seed: Optional random seed for reproducibility.
    :return: A NetworkX DiGraph representing the random DAG.
    """
    if seed is not None:
        random.seed(seed)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Generate a random ordering of the nodes
    # e.g. [2, 0, 3, 1, 4] for num_nodes=5
    node_order = list(range(num_nodes))
    random.shuffle(node_order)

    # Keep track of each node's out-degree count
    out_degrees = {node: 0 for node in node_order}

    # Ensure connectivity by always connecting each new node to
    # at least one of the existing nodes in an earlier position in node_order
    for i in range(1, num_nodes):
        current_node = node_order[i]
        # Pick at least one "parent" from among the nodes before current_node
        eligible_parents = node_order[:i]
        parent = random.choice(eligible_parents)
        G.add_edge(parent, current_node)
        out_degrees[parent] += 1

    # Now, optionally add extra edges from earlier nodes to current_node,
    # respecting max_out_degree and ensuring no cycles.
    for i in range(num_nodes):
        current_node = node_order[i]
        # Try to add edges from current_node to nodes *after* it in the order
        for j in range(i + 1, num_nodes):
            next_node = node_order[j]
            # Check out-degree limit
            if out_degrees[current_node] < max_out_degree:
                # Randomly decide if we add an edge
                # (You can change the probability or logic as needed)
                if random.random() < 1/num_nodes:
                    G.add_edge(current_node, next_node)
                    out_degrees[current_node] += 1
            else:
                # We've reached max out-degree, no more edges from current_node
                break

    return G



def draw_task_precedence_graph():
    # Initialize the graph
    G = nx.DiGraph()

    # Define the nodes and their positions
    positions = {
        5: (0, 2),
        7: (0, 1),
        6: (0, 0),
        8: (1, 1),
        4: (1, 2),
        2: (3.5, 2),
        1: (3, 1),
        3: (3.5, 0),
        9: (4, 1),
        10: (5, 1),
    }

    # Add nodes to the graph
    G.add_nodes_from(positions.keys())

    # Define edges
    edges = [
        (5, 7), (6, 7), (7, 8), (4, 8), (8, 2),
        (1, 2), (9,2), (10,2), (8, 3), (1, 3), (9,3),(10,3)
    ]

    # Add edges to the graph
    G.add_edges_from(edges)

    # Draw the graph
    plt.figure(figsize=(6, 4))
    nx.draw(
        G,
        pos=positions,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_weight="bold",
        arrowsize=20, # Make arrows bigger
        edgecolors="black"
    )
    plt.show()



def plot_result(n,m,w,t,x, R, CT):
    # 1) Extract Gurobi solutions
    solution_x = np.zeros((n, m, 2), dtype=int)
    for i in range(n):
        for k in range(m):
            for l in range(2):
                solution_x[i, k, l] = int(round(x[i, k, l].X))

    solution_R = np.zeros((n, 2), dtype=int)  # for destructive vs non-destructive
    for i in range(n):
        for r in range(2):
            solution_R[i, r] = int(round(R[i, r].X))

    solution_w = [w[i].X for i in range(n)]  # start times
    solution_t = [t[i].X for i in range(n)]  # durations

    # 2) Build a list of (task_id, side, start, duration, color)
    assignments = []
    for i in range(n):
        # find which side l is chosen for this task
        sides_for_task = [
            l for k in range(m) for l in range(2)
            if solution_x[i,k,l] == 1
        ]
        if len(sides_for_task) != 1:
            continue
        side_assigned = sides_for_task[0]  # 0 or 1

        start_time = solution_w[i]
        duration   = solution_t[i]

        # Decide on color based on destructive vs. non-destructive
        if solution_R[i,1] == 1:
            color_str = "#ff7f7f"  # pale red
            mode_str  = "destructive"
        else:
            color_str = "#6edb6e"  # pastel green
            mode_str  = "non-destructive"

        assignments.append({
            "task_id": i+1,
            "side": side_assigned,
            "start": start_time,
            "duration": duration,
            "color": color_str,
            "mode": mode_str
        })

    # 3) Create a figure (with white background) and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Explicitly set backgrounds to white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 4) Plot each task
    for task in assignments:
        tid      = task["task_id"]
        side     = task["side"]         # 0 => Left, 1 => Right
        start    = task["start"]
        duration = task["duration"]
        color    = task["color"]

        # side=0 => y=0, side=1 => y=1
        y_val = side

        ax.barh(
            y=y_val,
            width=duration,
            left=start,
            height=0.4,
            color=color,
            edgecolor="black"
        )

        # Label each bar with "Task i" in the middle
        ax.text(
            x=start + 0.5*duration,
            y=y_val,
            s=f"Task {tid}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    # 5) Beautify axes
    ax.set_ylim(-0.5, 1.5)

    # Label the two rows for side=0 (top) and side=1 (bottom)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Left side (l=0)", "Right side (l=1)"])
    ax.invert_yaxis()  # so side=0 is at the top

    ax.set_xlabel("Time")
    ax.set_title("Tasks on Left (top) and Right (bottom) Sides")

    # 6) Draw vertical lines every CT seconds and label them as stations
    # 6a) Find max finishing time so we know how many station blocks to draw
    max_finish = max(a["start"] + a["duration"] for a in assignments) if assignments else 0
    num_blocks = int(math.ceil(max_finish / CT))

    for block_index in range(1, num_blocks + 1):
        x_pos = block_index * CT
        # vertical line
        ax.axvline(x=x_pos, color="gray", linestyle="--", linewidth=1)

        # label station in the middle of each block
        # e.g., block 1 extends from x=0..CT, block 2 from x=CT..2*CT, etc.
        left_boundary = (block_index - 1)*CT
        right_boundary = block_index*CT
        mid_x = 0.5*(left_boundary + right_boundary)

        # We'll place the label above the bars
        # or just below the x-axis if you prefer. Let's put it near y=-0.3 for clarity:
        ax.text(
            x=mid_x,
            y=-0.3,
            s=f"Station {block_index}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold"
        )


    plt.show()
