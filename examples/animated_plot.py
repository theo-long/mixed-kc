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


def indent_lines(lines, spaces, postfix=""):
    indent_str = " " * spaces
    res = []
    for i, (text, is_hl) in enumerate(lines):
        if i == len(lines) - 1:
            res.append((f"{indent_str}{text}{postfix}", is_hl))
        else:
            res.append((f"{indent_str}{text}", is_hl))
    return res


def format_ast_lines(node, evaluated_map, current_node_id):
    if node is None:
        return []
    if node.get("type") == "primitive":
        return [(str(node.get("val")), False)]

    nid = node.get("id")
    if nid in evaluated_map:
        is_hl = nid == current_node_id
        return [(str(evaluated_map[nid]), is_hl)]

    is_hl = nid == current_node_id
    t = node.get("type")

    if t == "Var":
        return [(f"Var('{node.get('var', '')}')", is_hl)]
    elif t == "Const":
        return [(f"Const({node.get('val', '')})", is_hl)]
    elif t == "Flip":
        prob_lines = format_ast_lines(node.get("prob"), evaluated_map, current_node_id)
        if len(prob_lines) == 1:
            return [(f"Flip({prob_lines[0][0]})", is_hl or prob_lines[0][1])]
        else:
            return [("Flip(", is_hl)] + indent_lines(prob_lines, 2) + [(")", is_hl)]

    elif t == "Let":
        res = [("Let(", is_hl)]
        res.append((f"  '{node.get('var', '')}',", is_hl))

        bind_lines = format_ast_lines(
            node.get("binding"), evaluated_map, current_node_id
        )
        if len(bind_lines) == 1:
            res.append((f"  {bind_lines[0][0]},", bind_lines[0][1]))
        else:
            res.extend(indent_lines(bind_lines, 2, postfix=","))

        body_lines = format_ast_lines(node.get("body"), evaluated_map, current_node_id)
        if len(body_lines) == 1:
            res.append((f"  {body_lines[0][0]}", body_lines[0][1]))
        else:
            res.extend(indent_lines(body_lines, 2))

        res.append((")", is_hl))
        return res

    elif t == "IfThenElse":
        res = [("IfThenElse(", is_hl)]

        cond_lines = format_ast_lines(node.get("cond"), evaluated_map, current_node_id)
        if len(cond_lines) == 1:
            res.append((f"  {cond_lines[0][0]},", cond_lines[0][1]))
        else:
            res.extend(indent_lines(cond_lines, 2, postfix=","))

        then_lines = format_ast_lines(
            node.get("then_expr"), evaluated_map, current_node_id
        )
        if len(then_lines) == 1:
            res.append((f"  {then_lines[0][0]},", then_lines[0][1]))
        else:
            res.extend(indent_lines(then_lines, 2, postfix=","))

        else_lines = format_ast_lines(
            node.get("else_expr"), evaluated_map, current_node_id
        )
        if len(else_lines) == 1:
            res.append((f"  {else_lines[0][0]}", else_lines[0][1]))
        else:
            res.extend(indent_lines(else_lines, 2))

        res.append((")", is_hl))
        return res

    elif t == "Eq":
        res = [("Eq(", is_hl)]
        l_lines = format_ast_lines(node.get("left"), evaluated_map, current_node_id)
        if len(l_lines) == 1:
            res.append((f"  {l_lines[0][0]},", l_lines[0][1]))
        else:
            res.extend(indent_lines(l_lines, 2, postfix=","))
        r_lines = format_ast_lines(node.get("right"), evaluated_map, current_node_id)
        if len(r_lines) == 1:
            res.append((f"  {r_lines[0][0]}", r_lines[0][1]))
        else:
            res.extend(indent_lines(r_lines, 2))
        res.append((")", is_hl))
        return res

    elif t == "Observe":
        res = [("Observe(", is_hl)]
        c_lines = format_ast_lines(node.get("cond"), evaluated_map, current_node_id)
        if len(c_lines) == 1:
            res.append((f"  {c_lines[0][0]}", c_lines[0][1]))
        else:
            res.extend(indent_lines(c_lines, 2))
        res.append((")", is_hl))
        return res

    else:
        return [(f"<{node.get('type', 'Unknown')}_node>", is_hl)]


def create_animation(name: Literal["discrete", "gaussian"]):
    with open(f"visualize_data_{name}.json", "r") as f:
        data = json.load(f)

    comp_steps = data["compilation_steps"]
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

    initial_ast = data.get("initial_ast")
    evaluated_map = {}

    frames_data = []

    # Track visible graph
    visible_bdd_nodes = set()

    fig, (ax_ast, ax_graph) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]}
    )
    plt.subplots_adjust(bottom=0.15)

    def draw_current_state(frame_data):
        ax_ast.clear()
        ax_graph.clear()

        ax_ast.axis("off")
        ax_graph.axis("off")

        ast_str = frame_data["ast_str"]
        highlighting_ast_node = frame_data["highlighting_ast_node"]
        highlighting_bdd_node = frame_data["highlighting_bdd_node"]
        cur_visible_bdd_nodes = frame_data["visible_bdd_nodes"]
        cur_evaluated_map = frame_data["evaluated_map"]

        fig.suptitle("Knowledge Compilation Phase", fontsize=16)

        # Draw AST Text
        if initial_ast:
            lines = format_ast_lines(
                initial_ast, cur_evaluated_map, highlighting_ast_node
            )
            y_pos = 0.95
            line_height = 0.03
            ax_ast.set_title("IR Compilation State")
            for text, is_hl in lines:
                if is_hl:
                    ax_ast.text(
                        0.05,
                        y_pos,
                        text,
                        fontsize=12,
                        family="monospace",
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="none"),
                    )
                else:
                    ax_ast.text(
                        0.05,
                        y_pos,
                        text,
                        fontsize=12,
                        family="monospace",
                        va="top",
                        ha="left",
                    )
                y_pos -= line_height
        else:
            wrapped_ast = ast_str
            if len(ast_str) > 60:
                wrapped_ast = ast_str[:57] + "..."
            ax_ast.text(
                0.05,
                0.5,
                wrapped_ast,
                fontsize=12,
                family="monospace",
                va="center",
                ha="left",
                wrap=True,
            )
            ax_ast.set_title("Current Compilation Expr")

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
                node_color="skyblue",
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

        # No more WMC strings here

    # Step 1: Compilation Phase
    for step in comp_steps:
        expr_str = step.get("expr_repr", "")
        ast_id = step.get("ast_id")

        if step["event"] == "end_kc":
            res_val = step.get("result", "")
            if "EnumResult" in res_val:
                res_val = "EnumRes"

            if ast_id is not None:
                evaluated_map = dict(evaluated_map)
                evaluated_map[ast_id] = res_val

            nid = step.get("result_node")
            live_nids = step.get("live_node_ids")
            if live_nids is not None:
                new_visible = set()
                for lnid in live_nids:
                    if lnid in G_bdd.nodes:
                        new_visible.add(lnid)
                        new_visible.update(nx.descendants(G_bdd, lnid))
                visible_bdd_nodes = new_visible
            elif nid is not None and nid in G_bdd.nodes:
                visible_bdd_nodes = set(visible_bdd_nodes)
                visible_bdd_nodes.add(nid)
                visible_bdd_nodes.update(nx.descendants(G_bdd, nid))

                frames_data.append(
                    {
                        "ast_str": expr_str,
                        "highlighting_ast_node": ast_id,
                        "highlighting_bdd_node": nid if (nid in G_bdd.nodes) else None,
                        "visible_bdd_nodes": set(visible_bdd_nodes),
                        "evaluated_map": dict(evaluated_map),
                    }
                )

            frames_data.append(
                {
                    "ast_str": expr_str,
                    "highlighting_ast_node": ast_id,
                    "highlighting_bdd_node": None,
                    "visible_bdd_nodes": set(visible_bdd_nodes),
                    "evaluated_map": dict(evaluated_map),
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
        "--example", type=str, choices=["discrete", "gaussian"], default="discrete"
    )
    args = parser.parse_args()
    create_animation(args.example)
