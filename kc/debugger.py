from functools import wraps

import dd.autoref as _bdd

from kc.config import settings


class Debugger:
    def __init__(self):
        self.depth = 0

    def before_execute(self, node, env):
        indent = "  " * self.depth
        print(f"\n{indent}{node.__class__.__name__}(", end="")

        input(f"{indent}(Press Enter to evaluate...)")

        self.depth += 1

    def after_execute(self, node, result):
        self.depth -= 1
        indent = "  " * self.depth
        if isinstance(result, _bdd.Function):
            result_str = result.to_expr()
        else:
            result_str = str(result)
        print(f") ===> {result_str},\n{indent}", end="")


# A global debugger instance for simplicity,
# though you could also pass it inside the `env` dictionary.
DEBUGGER = Debugger()


def step_hook(func):
    @wraps(func)
    def wrapper(self, env, *args, **kwargs):
        if not settings.debug:
            return func(self, env, *args, **kwargs)

        # 1. Trigger the 'before' hook
        DEBUGGER.before_execute(self, env)

        # 2. Execute the actual interpret method
        result = func(self, env, *args, **kwargs)

        # 3. Trigger the 'after' hook
        DEBUGGER.after_execute(self, result)

        return result

    return wrapper
