import itertools
import pickle
import math
import random
import time
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass


def update_longest_paths_with_path(path: list[str]) -> None:
    """
    Update the global dict longest_paths with a new path.
    This will update the longest path from any node in the path to the last node.
    Only paths with at least long_path_threshold edges will be considered.
    """
    if len(path) > long_path_threshold:
        for i in range(len(path) - long_path_threshold):
            node_pair = (path[i], path[-1])
            candidate_path = path[i:]
            if node_pair not in longest_paths or len(candidate_path) > len(longest_paths[node_pair][0]):
                longest_paths[node_pair] = [candidate_path]
            elif len(candidate_path) == len(longest_paths[node_pair][0]):
                longest_paths[node_pair].append(candidate_path)


def update_longest_paths_with_cycle(cycle: list[str]) -> None:
    """
    Update the global dict longest_paths with a new cycle.
    This will update the longest path between any pair of nodes in the cycle.
    Only paths with at least long_path_threshold edges will be considered.
    """
    L = len(cycle)
    if L > long_path_threshold:
        path = cycle + cycle
        for i in range(L):
            for j in range(long_path_threshold, L):
                node_pair = (path[i], path[i + j])
                candidate_path = path[i : i + j + 1]
                if node_pair not in longest_paths or len(candidate_path) > len(longest_paths[node_pair][0]):
                    longest_paths[node_pair] = [candidate_path]
                elif len(candidate_path) == len(longest_paths[node_pair][0]):
                    longest_paths[node_pair].append(candidate_path)


def beam_search(
    G: nx.DiGraph,
    in_degree_weight: float = 0.0,      # reward for in-degree at last node
    out_degree_weight: float = 0.0,     # reward for out-degree at last node
    imbalance_weight: float = 0.0,      # reward for |in-out| imbalance at last node
    backtrack_weight: float = 0.0,      # reward for distance to any earlier node
    close_weight: float = 0.0,          # reward if last node has a direct back-edge to the path
    beam_width: int = 512,
    per_node_cap: int = 8,
    time_limit: float = 10.0,
    seed: int = 42,
) -> list[str]:
    """
    Heuristic longest simple cycle finder in a strongly connected DiGraph G.
    Returns: list of nodes in the best cycle found (no repeated end node).
    """
    t0 = time.time()
    rng = random.Random(seed)
    start = rng.choice(list(G.nodes()))  # choose a random start node

    # ---- priority computation for a partial path ----
    def priority(path, pos_map):
        u = path[-1]

        # distance to earlier nodes
        dist_u = shortest_dists.get(u, {})
        dmin = min((dist_u[node] for node in pos_map if node in dist_u), default=math.inf)

        # closability: any direct back-edge to earlier node?
        back_edge = any(x in G[u] for x in pos_map if x != u)

        # all heuristic terms
        score = (
            in_degree_weight * G.in_degree(u)
            + out_degree_weight * G.out_degree(u)
            + imbalance_weight * abs(G.in_degree(u) - G.out_degree(u))
            + backtrack_weight * dmin
            + (close_weight if back_edge else 0.0)
        )

        # negate the score so that sorting puts best paths at the top
        return -score

    # ---- beam items: (priority, path, pos_map) ----
    # pos_map: node -> index in path
    init_path = [start]
    init_map = {start: 0}
    beam = [(priority(init_path, init_map), init_path, init_map)]

    best_cycle = []
    best_len = 0

    while beam and (time.time() - t0) <= time_limit:
        # expand current beam
        expansions = []

        for _, path, pos_map in beam:
            u = path[-1]

            # try all successors
            for v in G.successors(u):
                if v in pos_map:
                    # found a cycle path[i:] where i = pos_map[v]
                    i = pos_map[v]
                    update_longest_paths_with_cycle(path[i:])
                    cyc_len = len(path) - i
                    if cyc_len > best_len:
                        best_len = cyc_len
                        best_cycle = path[i:]
                else:
                    new_path = path + [v]
                    update_longest_paths_with_path(new_path)
                    new_map = dict(pos_map)
                    new_map[v] = len(path)
                    pr = priority(new_path, new_map)
                    expansions.append((pr, new_path, new_map))

        if not expansions:
            break

        # ---- diversity: keep up to per_node_cap per last-node ----
        by_last = defaultdict(list)
        for item in expansions:
            last = item[1][-1]
            by_last[last].append(item)

        capped = []
        for last, items in by_last.items():
            items.sort(key=lambda t: t[0])  # lower pr (i.e., higher score) first
            capped.extend(items[:per_node_cap])

        # ---- global prune to beam_width ----
        capped.sort(key=lambda t: t[0])
        beam = capped[:beam_width]

    return best_cycle


@dataclass
class ParamSpec:
    init_value: int | float = 0
    mutation_step: int | float = 0
    precision: int | float = 0
    min_value: int | float | None = None
    max_value: int | float | None = None

    def random(self, init=None, rng=None):
        if init is None:
            init = self.init_value
        rand = rng.random() if rng else random.random()
        value = init + (rand * 2 - 1) * self.mutation_step
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        value = round(round(value / self.precision) * self.precision, 3)
        return value


def evolutionary_search(
    G: nx.DiGraph,
    param_space: dict[str, ParamSpec],
    population_size: int,
    n_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_frac: float,
    seed: int = 42,
) -> tuple[list, dict[str, int | float]]:
    """
    Evolutionary search for hyperparameters of beam_search.

    Args:
        G (nx.DiGraph): the graph
        param_space (dict): maps parameter name to (init_value, step_size, "continuous"/"discrete")
        population_size (int): number of candidate parameter sets per generation
        n_generations (int): number of generations
        mutation_rate (float): probability of mutating a parameter
        elite_frac (float): fraction of top candidates kept each generation
        seed (int): RNG seed

    Returns:
        (best_cycle, best_params)
    """
    rng = random.Random(seed)

    # --- initialize population ---
    population = [{k: spec.init_value for k, spec in param_space.items()}]
    population += [{k: spec.random(rng=rng) for k, spec in param_space.items()} for _ in range(population_size - 1)]

    best_score = 0
    best_cycle = []
    best_params = None

    for gen in range(1, n_generations + 1):
        # --- beam search with current population ---
        scored_pop = []
        for indiv in population:
            indiv["seed"] = seed
            seed += 1
            cycle = beam_search(G, **indiv)
            print(f"Params: {indiv}, found cycle of length {len(cycle)}: {' '.join(cycle)}")
            score = len(cycle)
            scored_pop.append((score, indiv, cycle))

        # --- sort by score ---
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        cur_best_score, cur_best_params, cur_best_cycle = scored_pop[0]
        if cur_best_score > best_score:
            best_score, best_params, best_cycle = cur_best_score, cur_best_params.copy(), cur_best_cycle

        # --- print best result so far ---
        print()
        print(f"Gen {gen}: Best params: {best_params}, cycle length {best_score}: {' '.join(best_cycle)}")
        longest_path = max((paths[0] for paths in longest_paths.values()), key=len, default=[])
        print(f"Longest path has {len(longest_path)} nodes: {' '.join(longest_path)}")
        print()

        # --- selection ---
        n_elite = max(1, int(elite_frac * population_size))
        elites = [indiv for _, indiv, _ in scored_pop[:n_elite]]

        # --- reproduction, crossover, mutation ---
        new_population = elites.copy()
        while len(new_population) < population_size:
            # crossover or duplication
            if rng.random() < crossover_rate:
                parents = rng.choices(elites, k=2)
                child = {k: parents[0 if rng.random() < 0.5 else 1][k] for k in param_space}
            else:
                child = rng.choice(elites).copy()
            # mutation
            for k, spec in param_space.items():
                if rng.random() < mutation_rate:
                    child[k] = spec.random(init=child[k], rng=rng)
            new_population.append(child)
        population = new_population

    return best_cycle, best_params


if __name__ == "__main__":
    # Load graph data
    with open("../data/graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)

    connector = graph_data["connector"]
    edges = graph_data["edges"]
    G = graph_data["graph"]
    strong_comps = graph_data["strong_comps"]

    # Create a graph object of the largest strongly connected component.
    # We make a copy instead of a view, so that subsequent operations run faster.
    # NOTE: `subgraph` does not guarantee the order of nodes and edges, due to the use of sets.
    # To ensure determinism, set the environment variable PYTHONHASHSEED (e.g. to 42).
    S = G.subgraph(strong_comps[0]).copy()

    # Remove an NSFW edge
    S.remove_edge("\u5973", "\u88f8")

    # Shortest distances between all pairs of nodes (global variable)
    shortest_dists = dict(nx.all_pairs_shortest_path_length(S))

    # Longest paths that will be found between all pairs of nodes (global variable)
    # longest_paths[u, v] will be a list of all the longest paths from u to v
    longest_paths = {}
    long_path_threshold = 65  # paths have to be this long to be counted as long

    # Evolutionary search for longest cycle
    param_space = {
        "in_degree_weight": ParamSpec(0.0, 0.3, 0.1),
        "out_degree_weight": ParamSpec(0.0, 0.3, 0.1),
        "imbalance_weight": ParamSpec(0.0, 0.3, 0.1),
        "backtrack_weight": ParamSpec(0.0, 0.3, 0.1),
        "close_weight": ParamSpec(0.0, 0.3, 0.1),
        "beam_width": ParamSpec(512, 256, 128, min_value=128),
        "per_node_cap": ParamSpec(8, 4, 2, min_value=2),
    }

    best_cycle, best_params = evolutionary_search(
        S,
        param_space,
        population_size=12,
        n_generations=15,
        crossover_rate=0.0,
        mutation_rate=0.4,
        elite_frac=0.25,
        seed=42,
    )

    # Print result
    print()
    print("Longest cycle:")
    for i in range(len(best_cycle)):
        term1 = best_cycle[i]
        term2 = best_cycle[(i + 1) % len(best_cycle)]
        print(f"{i + 1:3d}: {term1}{connector}{term2} ({', '.join(edges[term1, term2])})")
    print()
    print("Best params:", best_params)

    # Save graph data
    graph_data["longest_cycle"] = best_cycle  # list[str]: nodes in the cycle
    graph_data["best_params"] = best_params  # dict[str: int | float]
    graph_data["longest_paths"] = longest_paths  # dict[tuple[str, str], list[list[str]]]: (node, node) -> list of all longest paths

    with open("../data/graph_data.pkl", "wb") as f:
        pickle.dump(graph_data, f)
