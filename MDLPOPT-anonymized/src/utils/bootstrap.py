# ================= project bootstrap =================
import os
import sys
from pathlib import Path
import logging
import platform

logging.basicConfig(level=logging.INFO)

def get_project_root():
    """
    Resolve project root in a robust way.
    Priority:
    1. Environment variable THESIS_PROJECT_PATH
    2. Git root (if available)
    3. Parent directory containing 'src'
    """
    # 1. Explicit override (best for cluster)
    env_path = os.environ.get("THESIS_PROJECT_PATH")
    if env_path:
        root = Path(env_path).resolve()
        if root.exists():
            return root

    # 2. Git root (works locally & on cluster if repo cloned)
    try:
        import subprocess
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return Path(git_root)
    except Exception:
        pass

    # 3. Walk upwards until 'src' is found
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists():
            return parent

    raise RuntimeError(
        "Could not determine project root. "
        "Set THESIS_PROJECT_PATH environment variable."
    )


PROJECT_ROOT = get_project_root()
SRC_PATH = PROJECT_ROOT / "src"

# Add src to PYTHONPATH
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Change working directory to project root (optional)
os.chdir(PROJECT_ROOT)

logging.info(f"Project root resolved to: {PROJECT_ROOT}")
logging.info(f"Running on {platform.system()}")

# ================= end bootstrap =================
