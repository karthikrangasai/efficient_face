from pathlib import Path

EFFICIENT_FACE_DIR = Path(__file__).parent.absolute().resolve()
SRC_DIR = EFFICIENT_FACE_DIR.parent

ROOT_PATH = SRC_DIR.parent

CACHE_DIR = Path("~/.cache").expanduser().absolute()
DATA_DIR = CACHE_DIR / "ciFAIR"
