from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # dataset
    DATASET_MASK: str
    CSV_DELIMITER: str
    
    # clusters
    MAX_CLUSTERS: int
    CLUSTERS_DIR: str
    
    class Config:
        env_file = ".env"


settings = Settings()