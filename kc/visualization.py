from functools import wraps

import dd.autoref as _bdd

import kc.base
import kc.inference
import kc.model_count
import kc.real_values
import kc.terms
from kc.model_count import _model_count as original_model_count


class Tracker:
    def __init__(self):
        self.compilation_steps = []
        self.wmc_steps = []
        self.roots = []
        self._bdd = None
        self.start_kc_called = False
        self.final_weights = {}
        self.bdd_node_to_expr = {}
        self.all_nodes_info = {}
        self.all_nodes_visited = set()

    def start(self):
        self.active = True
        self.compilation_steps.clear()
        self.wmc_steps.clear()
        self.roots.clear()
        self.initial_ast = None
        self.start_kc_called = False
        self.final_weights.clear()
        self.bdd_node_to_expr.clear()
        self.all_nodes_info.clear()
        self.all_nodes_visited.clear()

    def stop(self):
        self.active = False


tracker = Tracker()


def ast_id(expr):
    return id(expr)


def format_weight(w):
    if type(w).__name__ == "ObservationWeights":
        if not w.scope:
            return str(round(w.likelihood, 2))
        return f"W({round(w.likelihood, 2)}, obs={len(w.scope)})"
    elif isinstance(w, (int, float)):
        return str(round(w, 2))
    return str(w)


def kc_hook_wrapper(orig_fn, cls):
    @wraps(orig_fn)
    def wrapper(self, env, state, *args, **kwargs):
        if not tracker.active:
            return orig_fn(self, env, state, *args, **kwargs)

        if not tracker.start_kc_called:
            tracker.initial_ast = extract_ast_tree(self)
            tracker.start_kc_called = True

        tracker._bdd = state.bdd
        if hasattr(state, "weights"):
            for k, v in state.weights.items():
                tracker.final_weights[k] = (format_weight(v[0]), format_weight(v[1]))

        expr_id = ast_id(self)
        tracker.compilation_steps.append(
            {
                "event": "start_kc",
                "ast_id": expr_id,
                "expr_repr": repr(self),
                "expr_type": type(self).__name__,
            }
        )

        res = orig_fn(self, env, state, *args, **kwargs)

        res_str = repr(res)
        res_node_id = None
        live_node_ids = []

        if isinstance(res, _bdd.Function):
            import gc

            res_node_id = abs(res.node)

            def add_node_rec(u, visited):
                if u is None:
                    return
                nid = abs(u.node)
                if nid in visited:
                    return
                visited.add(nid)
                if nid == 1:
                    tracker.all_nodes_info[nid] = {
                        "id": nid,
                        "var": "True" if u.node > 0 else "False",
                        "low": None,
                        "high": None,
                    }
                    return
                tracker.all_nodes_info[nid] = {
                    "id": nid,
                    "var": getattr(u, "var", str(nid)),
                    "low": abs(u.low.node) if u.low else None,
                    "high": abs(u.high.node) if u.high else None,
                }
                add_node_rec(u.low, visited)
                add_node_rec(u.high, visited)

            add_node_rec(res, tracker.all_nodes_visited)

            # GC check to find ONLY live python bdd references.
            live_node_ids = list(
                set(
                    [
                        abs(obj.node)
                        for obj in gc.get_objects()
                        if isinstance(obj, _bdd.Function)
                    ]
                )
            )

            if res.node == 1:
                res_str = "True"
            elif res.node == -1:
                res_str = "False"
            else:
                if res_node_id not in tracker.bdd_node_to_expr:
                    tracker.bdd_node_to_expr[res_node_id] = repr(self)
                expr_repr = tracker.bdd_node_to_expr[res_node_id]
                res_str = expr_repr if res.node > 0 else f"~({expr_repr})"
        elif hasattr(res, "enum_type"):
            res_str = (
                f"EnumResult(bits={[abs(b.node) for b in getattr(res, 'bits', [])]})"
            )

        tracker.compilation_steps.append(
            {
                "event": "end_kc",
                "ast_id": expr_id,
                "result": res_str,
                "result_node": res_node_id,
                "bdd_vars": list(state.bdd.vars),
                "live_node_ids": live_node_ids,
            }
        )
        return res

    return wrapper


def hooked_model_count(bdd, u, count, weights):
    if not tracker.active:
        return original_model_count(bdd, u, count, weights)

    res = original_model_count(bdd, u, count, weights)

    tracker.wmc_steps.append(
        {"node_idx": abs(u.node) if hasattr(u, "node") else str(u), "result": str(res)}
    )
    return res


_original_model_count_main = kc.model_count.model_count


def hooked_model_count_main(bdd, u, weights):
    if tracker.active:
        tracker.roots.append(u)
    return _original_model_count_main(bdd, u, weights)


_original_methods = []
_hooked = False


def apply_hooks():
    global _hooked
    if _hooked:
        return

    modules_to_check = [kc.terms, kc.real_values]
    for mod in modules_to_check:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, kc.base.PExpr):
                if "kc" in obj.__dict__:
                    orig = obj.__dict__["kc"]
                    _original_methods.append((obj, orig))
                    setattr(obj, "kc", kc_hook_wrapper(orig, obj))

    kc.model_count._model_count = hooked_model_count
    kc.model_count.model_count = hooked_model_count_main
    kc.inference.model_count = hooked_model_count_main
    _hooked = True


def remove_hooks():
    global _hooked
    if not _hooked:
        return
    for obj, orig in _original_methods:
        setattr(obj, "kc", orig)
    _original_methods.clear()
    kc.model_count._model_count = original_model_count
    kc.model_count.model_count = _original_model_count_main
    kc.inference.model_count = _original_model_count_main
    _hooked = False


def extract_ast_tree(node):
    if not isinstance(node, kc.base.PExpr):
        if hasattr(node, "val"):
            return {"type": "primitive", "val": node.val}
        return {"type": "primitive", "val": str(node)}
    nid = id(node)

    if isinstance(node, kc.terms.Let):
        return {
            "id": nid,
            "type": "Let",
            "var": node.var,
            "binding": extract_ast_tree(node.binding),
            "body": extract_ast_tree(node.body),
        }
    elif isinstance(node, kc.terms.IfThenElse):
        return {
            "id": nid,
            "type": "IfThenElse",
            "cond": extract_ast_tree(node.cond),
            "then_expr": extract_ast_tree(node.then_expr),
            "else_expr": extract_ast_tree(node.else_expr),
        }
    elif isinstance(node, kc.terms.Flip):
        prob = (
            extract_ast_tree(node.prob)
            if getattr(node, "prob", None) and isinstance(node.prob, kc.base.PExpr)
            else {"type": "primitive", "val": node.prob}
        )
        return {"id": nid, "type": "Flip", "prob": prob}
    elif isinstance(node, kc.terms.Var):
        return {"id": nid, "type": "Var", "var": node.var}
    elif isinstance(node, kc.terms.Const):
        return {"id": nid, "type": "Const", "val": node.val}
    elif isinstance(node, kc.terms.Observe):
        return {"id": nid, "type": "Observe", "cond": extract_ast_tree(node.cond)}
    elif hasattr(node, "left") and hasattr(node, "right"):  # Equality
        return {
            "id": nid,
            "type": "Eq",
            "left": extract_ast_tree(node.left),
            "right": extract_ast_tree(node.right),
        }
    else:
        return {"id": nid, "type": type(node).__name__, "repr": repr(node)}


def extract_bdd_dag(roots, final_weights):
    nodes = []

    # roots is no longer needed to traverse, we just map out from serialized all_nodes_info
    for node_id, info in tracker.all_nodes_info.items():
        var_name = info["var"]
        if node_id == 1:
            nodes.append(
                {
                    "id": 1,
                    "var": "True",
                    "low": None,
                    "high": None,
                    "low_weight": "",
                    "high_weight": "",
                }
            )
            nodes.append(
                {
                    "id": -1,
                    "var": "False",
                    "low": None,
                    "high": None,
                    "low_weight": "",
                    "high_weight": "",
                }
            )
            continue
        w_pos, w_neg = final_weights.get(var_name, ("", ""))

        nodes.append(
            {
                "id": info["id"],
                "var": info["var"],
                "low": info["low"],
                "high": info["high"],
                "low_weight": w_neg,
                "high_weight": w_pos,
            }
        )

    return nodes
