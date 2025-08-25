import json
import pickle
import re
import sys
import networkx as nx


def make_regex(connector: str, strict: bool) -> str:
    if strict:
        # Each term must be a single kanji
        term = r"([\u4e00-\u9fa5])"
    else:
        # Each term can be either a string of kanjis or katakanas,
        # possibly mixed with digits and English letters
        term = r"([0-9A-Za-z\u4e00-\u9fa5]+|[0-9A-Za-z\u30a0-\u30ff]+)"
    # Ignore secondary titles
    suffix = r"(\s*[(（～].*)?"
    return "^" + term + connector + term + suffix + "$"


def load_songs(filename: str, regex: str) -> dict[tuple[str, str], list[str]]:
    """
    Load all song information.
    Returns a dict where the key is a pair of two terms that
    can form a song's title when connected by the connector word,
    and the value is a list of artists who have sung a song with this title.
    """

    # Load song info from input file
    songs = []
    total = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            js = json.loads(line)
            if js.get("title") is None:
                continue
            songs.append(js)

    print(f"{total} songs scraped")
    print(f"{len(songs)} songs valid")

    # Match all song info against regex
    def match_regex(song):
        match = re.match(regex, song["title"])
        if match:
            song["term1"] = match.group(1)
            song["term2"] = match.group(2)
        return bool(match)

    songs = [song for song in songs if match_regex(song)]
    songs = sorted(songs, key=lambda song: song["id"])
    print(f"{len(songs)} songs match regex")

    # Dedup song titles
    edges = {}
    for song in songs:
        pair = song["term1"], song["term2"]
        edges.setdefault(pair, []).append(song["artist"])
    print(f"{len(edges)} songs after dedup")
    return edges


def display_comps_info(
    G: nx.DiGraph,
    components: list[set[str]],
    adj: str,  # "weak" or "strong"
) -> list[list[str]]:
    """
    Display information about the connected components of a graph.
    Also sort the components by size, and the nodes in each component by (out_degree - in_degree).
    """
    print()
    print(f"{len(components)} {adj}ly connected components:")
    components = [list(comp) for comp in components]
    for comp in components:
        subG = G.subgraph(comp)
        # Sort the nodes so that nodes with more out-degrees than in-degrees come in front.
        # In case of a tie, sort the nodes by Unicode.
        comp.sort(key=lambda node: (subG.in_degree(node) - subG.out_degree(node), node))
    # Sort the components by size in descending order. Break ties by unicode of the nodes.
    components.sort(key=lambda comp: (-len(comp), comp))
    for i, comp in enumerate(components):
        subG = G.subgraph(comp)
        print(f"  Component {i}: {subG}, nodes: {' '.join(comp)}")
    return components


if __name__ == "__main__":
    # Create regex and load songs
    connector = sys.argv[1] if len(sys.argv) > 1 else "の"
    strict = sys.argv[2] == "True" if len(sys.argv) > 2 else True
    regex = make_regex(connector=connector, strict=strict)
    edges = load_songs("../data/songs.jsonl", regex)

    # Create a directed graph with the given edges
    G = nx.DiGraph()
    G.add_edges_from(edges.keys())
    print(f"{len(G.nodes())} distinct terms")

    # Find weakly connected components
    weak_comps = list(nx.weakly_connected_components(G))
    weak_comps = display_comps_info(G, weak_comps, "weak")

    # Find strongly connected components
    strong_comps = nx.strongly_connected_components(G)
    strong_comps = [comp for comp in strong_comps if len(comp) > 1]  # ignore trivial components
    strong_comps = display_comps_info(G, strong_comps, "strong")

    graph_data = {
        "connector": connector,  # str
        "edges": edges,  # dict[tuple[str, str], list[str]]: (node, node) -> list of artists
        "graph": G,  # nx.DiGraph
        "weak_comps": weak_comps,  # list[list[str]]: each list contains the nodes in one weakly connected component
        "strong_comps": strong_comps,  # list[list[str]]: each list contains the nodes in one strongly connected component
    }
    with open("../data/graph_data.pkl", "wb") as f:
        pickle.dump(graph_data, f)
