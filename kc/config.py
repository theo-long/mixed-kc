from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_implicit_flags=True)

    debug: bool = False
    single_observe_eps: bool = False
    transform_measures: bool = False


settings = Config()
print("---Settings---")
print(settings.model_dump_json(indent=2))
print("--------------")
