from functools import wraps

from kc.config import settings

class Debugger:
    def __init__(self):
        self.depth = 0

    def before_execute(self, node, env):
        indent = "  " * self.depth
        print(
            f"{indent}▶ Stepping into: {node.__class__.__name__} | Node State: {vars(node)}"
        )

        input(f"{indent}(Press Enter to evaluate...)")

        self.depth += 1

    def after_execute(self, node, result):
        self.depth -= 1
        indent = "  " * self.depth
        print(f"{indent}◀ Evaluated: {node.__class__.__name__} -> {result}\n")


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
