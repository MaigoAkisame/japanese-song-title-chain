import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def ellipse_points_arclen(
    n: int,
    a: float = 1.0,
    b: float = 0.5,
    theta0: float = 0.0,
    oversample: int = 2000,
) -> NDArray:
    """
    Return n points approximately equally spaced by arc length
    on the ellipse x=a cosθ, y=b sinθ.

    Args:
        n (int): number of points.
        a, b (float): semi-axes.
        theta0 (float): rotation (radians) of the first point.
        oversample (int): refinement factor for the arc-length table.
                          Higher = more accurate, slower.

    Returns:
        A numpy array of shape (n, 2), standing for the coordinates of each point.
    """
    # Dense parameter grid over [0, 2π]
    m = n * oversample
    thetas = np.linspace(0, 2*np.pi, m+1)
    # Speed along ellipse: |r'(θ)| = sqrt( (−a sinθ)^2 + (b cosθ)^2 )
    speed = np.sqrt((a*np.sin(thetas))**2 + (b*np.cos(thetas))**2)

    # Cumulative arc length via trapezoid rule
    # s(θ_k) ≈ ∑ 0..k-1 0.5*(speed[k]+speed[k+1]) * Δθ
    dtheta = thetas[1] - thetas[0]
    ds = 0.5 * (speed[:-1] + speed[1:]) * dtheta
    s_cum = np.concatenate(([0.0], np.cumsum(ds)))
    total_len = s_cum[-1]

    # Target arc-lengths (shifted by theta0)
    s_targets = (np.arange(n) * total_len / n) % total_len

    # If theta0 != 0, map it to its arc length offset and add
    if theta0 != 0.0:
        # Find arc-length at theta0 by interpolation
        # (wrap theta0 into [0, 2π))
        t0 = (theta0 % (2*np.pi))
        s0 = np.interp(t0, thetas, s_cum)
        s_targets = (s_targets + s0) % total_len

    # Invert s(θ) with interpolation to get θ_k
    thetas_eq = np.interp(s_targets, s_cum, thetas)  # monotonically increasing

    # Points on ellipse
    # Negate the x coordinate so the points are arranged clockwise
    return np.vstack([-a * np.cos(thetas_eq), b * np.sin(thetas_eq)]).T


if __name__ == "__main__":
    # Load graph data
    with open("../data/graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)

    connector = graph_data["connector"]
    edges = graph_data["edges"]
    G = graph_data["graph"]
    weak_comps = graph_data["weak_comps"]
    strong_comps = graph_data["strong_comps"]
    longest_between = graph_data["longest_paths"]

    # Construct the largest weakly and strongly connected components
    W = G.subgraph(weak_comps[0]).copy()
    S = G.subgraph(strong_comps[0]).copy()

    # Remove edges in S from W
    W.remove_edges_from(S.edges())

    # Remove self-loops in W
    edges_to_remove = [(u, v) for u, v in W.edges() if u == v]
    print(f"Removing self-loops: {edges_to_remove}")
    W.remove_edges_from(edges_to_remove)

    # Verify that the residual graph is acyclic
    assert nx.is_directed_acyclic_graph(W)

    # In the residual graph, for each node X in S, find the longest path to and from X
    topo = list(nx.topological_sort(W))
    longest_to = {node: [[node]] for node in W.nodes()}
    longest_from = {node: [[node]] for node in W.nodes()}
    for u in topo:
        for v in W.successors(u):
            if len(longest_to[u][0]) + 1 > len(longest_to[v][0]):
                longest_to[v] = [path + [v] for path in longest_to[u]]
            elif len(longest_to[u][0]) + 1 == len(longest_to[v][0]):
                longest_to[v] += [path + [v] for path in longest_to[u]]
    for v in reversed(topo):
        for u in W.predecessors(v):
            if len(longest_from[v][0]) + 1 > len(longest_from[u][0]):
                longest_from[u] = [[u] + path for path in longest_from[v]]
            elif len(longest_from[v][0]) + 1 == len(longest_from[u][0]):
                longest_from[u] += [[u] + path for path in longest_from[v]]

    # Find overall longest path(s)
    max_path_len = 0
    best_pairs = []
    for (u, v) in longest_between:
        path_len = len(longest_to[u][0]) + len(longest_between[u, v][0]) + len(longest_from[v][0]) - 2
        if path_len > max_path_len:
            max_path_len = path_len
            best_pairs = [(u, v)]
        elif path_len == max_path_len:
            best_pairs.append((u, v))

    best_paths = [
        path_to_u + path_between[1:-1] + path_from_v
        for u, v in best_pairs
        for path_to_u in longest_to[u]
        for path_between in longest_between[u, v]
        for path_from_v in longest_from[v]
    ]

    # Print longest path(s)
    print()
    print(f"Found {len(best_paths)} longest path(s) of {len(best_paths[0])} nodes and {len(best_paths[0]) - 1} edges")
    for k, path in enumerate(best_paths, 1):
        print()
        print(f"Path #{k}:")
        for i in range(len(path) - 1):
            term1, term2 = path[i], path[i + 1]
            print(f"    {i + 1:3d}: {term1}{connector}{term2} ({', '.join(edges[term1, term2])})")

    # FROM HERE BELOW: VISUALIZE LONGEST PATHS
    # The code is based on the knowledge that the graph formed by the longest paths consists of:
    # - A main cycle with alternatives at a few nodes;
    # - A few in-coming and out-going limbs.

    # Create a graph with all the best paths
    G = nx.DiGraph()
    for path in best_paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])

    # Add two edges that I've found manually but are missing in the results
    G.add_edge("雨", "港")
    G.add_edge("港", "月")

    # Find the main cycle
    main_cycle = max((longest_between[pair][0] for pair in best_pairs), key=len)
    # Draw the main cycle on an ellipse, with points spaced evenly
    a, b = 1.0, 0.5  # parameters of the ellipse
    coords = ellipse_points_arclen(len(main_cycle), a=a, b=b, theta0=0.0, oversample=2000)  # numpy array of shape (n, 2)
    coords = dict(zip(main_cycle, coords))  # turn into a dict that maps each node to its coordinates

    # Draw the alternatives on the main cycle
    for node in G.nodes():
        if node not in coords:
            u = list(G.predecessors(node))
            v = list(G.successors(node))
            if len(u) > 0 and len(v) > 0 and u[0] in main_cycle and v[0] in main_cycle:
                i = main_cycle.index(u[0])
                rival = main_cycle[(i + 1) % len(main_cycle)]
                coords[node] = coords[rival] * 0.95
                coords[rival] *= 1.05

    # Draw the in-coming limbs
    for u, v in best_pairs:
        for path in longest_to[u]:
            coord = coords[u]
            direction = coord / np.linalg.norm(coord) * 0.12
            for node in path[-2::-1]:
                coords[node] = coord = coord + direction

    # Hard code the coordinates of the out-going limbs
    coords["石"] = np.array([0.0, 0.0])
    coords["舟"] = np.array([0.0, -0.1])
    coords["人"] = np.array([0.0, -0.2])
    coords["瀬"] = np.array([-0.1, -0.3])
    coords["力"] = np.array([0.1, -0.3])

    # Create and save a figure
    plt.figure(figsize=(20, 10))
    nx.draw_networkx_edges(G, coords, alpha=0.5)
    nx.draw_networkx_nodes(G, coords, node_color="#FFFF80", edgecolors="black")
    nx.draw_networkx_labels(G, coords, font_family="MS Gothic", font_size=12)  # font supports kanji
    plt.axis("off")
    plt.savefig(f"../data/longest_paths.png", dpi=100, bbox_inches="tight")
