import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_comps(
    G: nx.DiGraph,
    components: list[set[str]],
    filename: str,
    figsize: tuple[float, float],
    node_coords: dict[str, tuple[float, float]] | None = None,  # position of each node
) -> dict[str, tuple[float, float]]:
    """
    Visualize connected components of a graph.
    """
    n = len(components)
    U = nx.DiGraph()  # union of all components
    for i, comp in enumerate(components):
        U = nx.union(U, G.subgraph(comp))

    # Compute an ideal position for each node, if not provided
    if node_coords is None:
        node_coords = {}  # position of each node
        for i, comp in enumerate(components):
            subG = G.subgraph(comp)
            comp_coords = nx.nx_agraph.graphviz_layout(subG, prog="neato", args="-Goverlap=false")

            # Scale the node positions in this component
            if i == 0:  # main component
                center = (0, 0)
                scale = 10
            else:
                center = (13, 9 * ((i - 1) / (n - 2) * 2 - 1))
                scale = 1

            coords = np.array(list(comp_coords.values()))
            min_coords, max_coords = coords.min(axis=0) - 1e-3, coords.max(axis=0) + 1e-3
            coords = (coords - min_coords) / (max_coords - min_coords) * 2 - 1
            coords = np.tanh(1.3 * coords) / np.tanh(1.3)  # non-linear transform to enlarge the center part
            coords = coords * scale + np.array(center)
            for node, coord in zip(comp_coords.keys(), coords):
                node_coords[node] = tuple(coord)

    # Color-code each node: nodes with more out-degrees are greener; nodes with more in-degrees are redder
    nodes = U.nodes()
    degree_diffs = [U.in_degree(node) - U.out_degree(node) for node in nodes]
    vmin, vmax = min(degree_diffs) - 1e-6, max(degree_diffs) + 1e-6
    cmap = plt.cm.RdYlGn
    node_colors = [cmap(1 - (diff - vmin) / (vmax - vmin)) for diff in degree_diffs]

    # Use random colors for edges
    np.random.seed(42)
    edge_colors = np.random.uniform(0.5, 0.75, size=(len(U.edges()), 3))

    # Make a plot with matplotlib
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(U, node_coords, edge_color=edge_colors, alpha=0.5)
    nx.draw_networkx_nodes(U, node_coords, nodelist=nodes, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(U, node_coords, font_family="MS Gothic", font_size=12)  # font supports kanji
    plt.axis("off")

    # Save plot to a file
    plt.savefig(filename, dpi=100, bbox_inches="tight")

    # Return the coordinates of each node
    return node_coords


if __name__ == "__main__":
    # Load graph data
    with open("../data/graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)

    G = graph_data["graph"]
    weak_comps = graph_data["weak_comps"]
    strong_comps = graph_data["strong_comps"]

    # Visualize weakly connected components, and remember the node coordinates
    node_coords = plot_comps(G, weak_comps, "../data/weak_comps.png", figsize=(28, 21))

    # Visualize strongly connected components
    plot_comps(G, strong_comps, "../data/strong_comps.png", figsize=(16, 12), node_coords=node_coords)

    # Save the node coordinates into graph_data.pkl
    graph_data["node_coords"] = node_coords  # dict[str, tuple[float, float]]: node -> coordinates
    with open("../data/graph_data.pkl", "wb") as f:
        pickle.dump(graph_data, f)
