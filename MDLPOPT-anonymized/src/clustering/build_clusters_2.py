# #from pathlib import Path
# import logging
# import os
# import sys
# from pathlib import Path

# PROJECT_PATH = Path(os.getcwd())
# os.chdir(PROJECT_PATH)
# sys.path.append(os.path.join(PROJECT_PATH, "src"))

# from utils.bootstrap import get_project_root
# PROJECT_ROOT = get_project_root()

# ================= project bootstrap =================
import os
import sys
from pathlib import Path
import logging
import platform

logging.basicConfig(level=logging.INFO)

def get_project_root():
    """
    Resolve project root robustly.
    Priority:
    1. THESIS_PROJECT_PATH env var
    2. Git root
    3. Folder named 'MDLPOPT'
    """
    # 1. Explicit override
    env_path = os.environ.get("THESIS_PROJECT_PATH")
    if env_path:
        root = Path(env_path).resolve()
        if (root / "src").exists():
            return root

    # 2. Git root (BEST if repo is cloned normally)
    try:
        import subprocess
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return Path(git_root)
    except Exception:
        pass

    # 3. Walk upwards and look specifically for MDLPOPT
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "MDLPOPT" and (parent / "src").exists():
            return parent

    raise RuntimeError(
        "Could not determine project root. "
        "Set THESIS_PROJECT_PATH explicitly."
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

from clustering.clustering_pain_trajectories import PainTrajectoryClustering


clusterer = PainTrajectoryClustering(n_clusters_list=range(2, 26))
clusterer.load_dataframe()
clusterer.extract_trajectories()

results_left = clusterer.run_clustering_experiments(side="left", methods=["kmeans", "dtw", "kshape", "fpca", "autoencoder"])
results_right = clusterer.run_clustering_experiments(side="right", methods=["kmeans", "dtw", "kshape", "fpca", "autoencoder"])

silhouette_scores_L, inertias_L = clusterer.evaluate_optimal_clusters(side="left")
silhouette_scores_R, inertias_R = clusterer.evaluate_optimal_clusters(side="right")