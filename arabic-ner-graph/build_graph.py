import argparse
import networkx as nx
from ner_rule_based import build_nlp
from relation_extraction import extract_all, DEMO_TEXT


COLOR_MAP = {
    "PER": "#D85A30",
    "PERS": "#D85A30",
    "ORG": "#7F77DD",
    "LOC": "#1D9E75",
    "MISC": "#888780",
}


def build_graph(relations) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for e1_text, e1_type, rel, e2_text, e2_type in relations:
        if not G.has_node(e1_text):
            G.add_node(e1_text, type=e1_type)
        if not G.has_node(e2_text):
            G.add_node(e2_text, type=e2_type)
        G.add_edge(e1_text, e2_text, label=rel)
    return G


def render_pyvis(G: nx.MultiDiGraph, out_path: str = "graph.html"):
    from pyvis.network import Network

    net = Network(directed=True, notebook=False, cdn_resources="in_line")
    for node, attrs in G.nodes(data=True):
        color = COLOR_MAP.get(attrs.get("type"), "#888888")
        net.add_node(node, label=node, color=color, title=attrs.get("type", ""))
    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, label=attrs.get("label", ""))

    net.show(out_path, notebook=False)
    print(f"Graph written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build and visualize the entity relation graph")
    parser.add_argument("--file", type=str, default=None, help="Arabic text file to analyze")
    parser.add_argument("--out", type=str, default="graph.html", help="Output HTML path")
    parser.add_argument("--export", type=str, default=None, help="Also export to .graphml or .gexf")
    args = parser.parse_args()

    
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEMO_TEXT

    nlp = build_nlp()
    doc = nlp(text)
    relations = extract_all(doc)

    if not relations:
        print("No relations extracted — nothing to graph.")
        return

    G = build_graph(relations)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    render_pyvis(G, args.out)

    #if args.export:
    if args.export.endswith(".graphml"):
            nx.write_graphml(G, args.export)
            print(f"Exported graph structure to {args.export}")
            
""" 
elif args.export.endswith(".gexf"):
    nx.write_gexf(G, args.export)
    else:
    print("Unsupported export format — use .graphml or .gexf")
 return 
 """
         


if __name__ == "__main__":
    main()
