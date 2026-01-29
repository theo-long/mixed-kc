import sys

from pydantic_settings import BaseSettings, SettingsConfigDict


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook or IPython environment."""
    return "ipykernel" in sys.modules


# Set _cli_parse_args to None if in a notebook, otherwise use default behavior (parse sys.argv)
cli_args_value = None if is_notebook() else True


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=cli_args_value, cli_implicit_flags=True
    )

    debug: bool = False
    single_observe_eps: bool = False
    transform_measures: bool = False


settings = Config()
print("---Settings---")
print(settings.model_dump_json(indent=2))
print("--------------")
