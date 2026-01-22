from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_implicit_flags=True)

    debug: bool = False


settings = Config()
