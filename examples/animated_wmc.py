from typing import Literal
import json

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button


def get_layout(nodes, edges, G):
    in_degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        in_degree[e[1]] += 1

    roots = [n["id"] for n in nodes if in_degree[n["id"]] == 0]
    layer_num = {}

    # DFS to find longest path from any root to node
    def dfs(curr, d):
        layer_num[curr] = max(layer_num.get(curr, 0), d)
        for e in edges:
            if e[0] == curr:
                dfs(e[1], d + 1)

    for r in roots:
        dfs(r, 0)

    for n in nodes:
        if n["id"] not in layer_num:
            layer_num[n["id"]] = 0

    max_layer = max(layer_num.values()) if layer_num else 0
    for n in nodes:
        if n.get("var") in ("True", "False"):
            layer_num[n["id"]] = max_layer + 1

    for n in nodes:
        G.nodes[n["id"]]["subset"] = layer_num[n["id"]]

    pos = nx.multipartite_layout(G, subset_key="subset", align="horizontal")
    # multipartite_layout aligns subsets along the y-axis if align='horizontal'.
    # v[1] is the layer coordinate, v[0] is the position within the layer.
    # We want roots at the top, so we negate y (v[1]).
    new_pos = {}
    for k, v in pos.items():
        new_pos[k] = (v[0], -v[1])
    return new_pos


# WMC Visualizer (no AST/IR display)


def create_animation(name: Literal["discrete", "gaussian"]):
    with open(f"visualize_data_{name}.json", "r") as f:
        data = json.load(f)

    wmc_steps = data["wmc_steps"]
    bdd_nodes = data["bdd_nodes"]

    G_bdd = nx.DiGraph()
    edges_bdd = []
    e_colors_bdd = {}
    edge_labels_bdd = {}
    for n in bdd_nodes:
        G_bdd.add_node(n["id"], label=str(n["var"]).lstrip("_"))
        if n["low"] is not None:
            edges_bdd.append((n["id"], n["low"]))
            e_colors_bdd[(n["id"], n["low"])] = "red"
            low_w = n.get("low_weight")
            if low_w:
                edge_labels_bdd[(n["id"], n["low"])] = low_w
        if n["high"] is not None:
            edges_bdd.append((n["id"], n["high"]))
            e_colors_bdd[(n["id"], n["high"])] = "green"
            high_w = n.get("high_weight")
            if high_w:
                edge_labels_bdd[(n["id"], n["high"])] = high_w

    G_bdd.add_edges_from(edges_bdd)

    if len(G_bdd.nodes) > 0:
        pos_bdd = get_layout(bdd_nodes, edges_bdd, G_bdd)
    else:
        pos_bdd = {}

    frames_data = []

    # Track visible graph
    visible_bdd_nodes = set()
    wmc_results = {}

    fig, ax_graph = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)

    def draw_current_state(frame_data):
        ax_graph.clear()
        ax_graph.axis("off")

        highlighting_bdd_node = frame_data["highlighting_bdd_node"]
        cur_visible_bdd_nodes = frame_data["visible_bdd_nodes"]
        cur_wmc_results = frame_data["wmc_results"]

        fig.suptitle("Weighted Model Counting Phase", fontsize=16)

        ax_graph.set_title("BDD State")
        ax_graph.margins(0.2)

        # Draw BDD Graph
        if cur_visible_bdd_nodes:
            subG = G_bdd.subgraph(list(cur_visible_bdd_nodes))
            nx.draw_networkx_nodes(
                subG,
                pos_bdd,
                ax=ax_graph,
                node_size=800,
                node_color="lightgray",
            )

            # Highlight
            if highlighting_bdd_node is not None and highlighting_bdd_node in pos_bdd:
                nx.draw_networkx_nodes(
                    G_bdd.subgraph([highlighting_bdd_node]),
                    pos_bdd,
                    ax=ax_graph,
                    node_size=800,
                    node_color="gold",
                )

            nx.draw_networkx_labels(
                subG,
                pos_bdd,
                ax=ax_graph,
                labels={n: G_bdd.nodes[n]["label"] for n in cur_visible_bdd_nodes},
                font_size=8,
            )

            drawn_edges = [
                e
                for e in edges_bdd
                if e[0] in cur_visible_bdd_nodes and e[1] in cur_visible_bdd_nodes
            ]
            if drawn_edges:
                edge_colors_list = [e_colors_bdd[e] for e in drawn_edges]
                nx.draw_networkx_edges(
                    G_bdd,
                    pos_bdd,
                    ax=ax_graph,
                    edgelist=drawn_edges,
                    edge_color=edge_colors_list,
                    arrows=True,
                    arrowsize=12,
                    connectionstyle="arc3,rad=0.1",
                )

                labels_to_draw = {
                    e: edge_labels_bdd[e] for e in drawn_edges if e in edge_labels_bdd
                }
                nx.draw_networkx_edge_labels(
                    G_bdd,
                    pos_bdd,
                    ax=ax_graph,
                    edge_labels=labels_to_draw,
                    font_size=7,
                    label_pos=0.3,
                    bbox=dict(alpha=0.0),
                )

        # Draw WMC strings next to nodes
        for nid, res_str in cur_wmc_results.items():
            if nid in pos_bdd:
                x, y = pos_bdd[nid]
                ax_graph.text(
                    x + 0.1,
                    y,
                    res_str,
                    color="blue",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

        # Compilation Phase skipped (fully compiled BDD is shown)

    visible_bdd_nodes = set([n["id"] for n in bdd_nodes])

    # Initial frame with full BDD
    frames_data.append(
        {
            "highlighting_bdd_node": None,
            "visible_bdd_nodes": set(visible_bdd_nodes),
            "wmc_results": {},
        }
    )

    # Step 2: WMC Phase
    for step in wmc_steps:
        nid_str = step["node_idx"]
        res_str = step["result"]
        try:
            nid = int(nid_str)
        except ValueError:
            continue

        wmc_results = dict(wmc_results)
        wmc_results[nid] = res_str

        frames_data.append(
            {
                "highlighting_bdd_node": nid,
                "visible_bdd_nodes": set(visible_bdd_nodes),
                "wmc_results": dict(wmc_results),
            }
        )

    current_frame = [0]

    def update_frame():
        draw_current_state(frames_data[current_frame[0]])
        fig.canvas.draw_idle()

    def next_frame(event):
        if current_frame[0] < len(frames_data) - 1:
            current_frame[0] += 1
            update_frame()

    def prev_frame(event):
        if current_frame[0] > 0:
            current_frame[0] -= 1
            update_frame()

    def key_press(event):
        if event.key == "right":
            next_frame(None)
        elif event.key == "left":
            prev_frame(None)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next ->")
    bprev = Button(axprev, "<- Prev")

    bnext.on_clicked(next_frame)
    bprev.on_clicked(prev_frame)
    fig.canvas.mpl_connect("key_press_event", key_press)

    if frames_data:
        update_frame()
        plt.show()
    else:
        print("No frames captured to animate.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example", type=str, default="discrete", choices=["discrete", "gaussian"]
    )
    args = parser.parse_args()
    create_animation(args.example)
