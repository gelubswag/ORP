from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # dataset
    DATASET_MASK: str
    CSV_DELIMITER: str
    CSV_NEWLINE: str
    HAS_HEADER: bool
    IGNORE_ATTRS: list

    class Config:
        env_file = ".env"


settings = Settings()
