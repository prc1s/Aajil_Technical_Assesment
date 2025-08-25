from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_file: Path
    csv_name: str

@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    source_file: Path
    csv_name: str
    column_names: list


@dataclass
class DataCategorisationConfig:
    root_dir: Path
    source: Path
    seeds: Path

    model_name: str
    batch_size: int
    normalise: bool

    use_pca: bool
    pca_n: int
    k_grid: type
    conf_threshold: float
    random_state: int
    tau: float

    experiment: str
    run_name: str