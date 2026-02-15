from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures"
DATA_DIR = PROJECT_ROOT / "TIMEVIEW" / "experiments" / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
