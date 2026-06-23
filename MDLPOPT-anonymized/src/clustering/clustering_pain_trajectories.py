import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging
from pathlib import Path
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#PROJECT_PATH=Path(os.environ["THESIS_PROJECT_PATH"])
PROJECT_PATH = Path(os.getcwd())
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PainTrajectoryClustering:
    def __init__(self,
                 data_path=PROJECT_PATH / "data" / "processed" / "CL_all_visits_df.csv",
                 save_root=PROJECT_PATH / "data" / "results" / "clustering",
                 figure_root=PROJECT_PATH / "data" / "results" / "figures",
                 n_clusters_list=range(2,25),
                 random_state=42,
                 device='cpu'):

        self.data_path = data_path
        self.save_root = save_root
        self.figure_root = figure_root
        self.n_clusters_list = n_clusters_list
        self.random_state = random_state
        self.device = device

        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.figure_root, exist_ok=True)

        self.df = None
        self.left_ts = None
        self.right_ts = None

    def load_dataframe(self):
        logging.info(f"Loading dataframe from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sort_values(["ID", "Visit"])
        return self.df

    def extract_trajectories(self):
        if self.df is None:
            self.load_dataframe()

        logging.info("Extracting left and right pain trajectories")
        pivot_left = self.df.pivot(index="ID", columns="Visit", values="VXXWOMKPL").dropna()
        pivot_right = self.df.pivot(index="ID", columns="Visit", values="VXXWOMKPR").dropna()

        scaler = TimeSeriesScalerMeanVariance()
        self.left_ts = scaler.fit_transform(pivot_left.to_numpy(dtype=float))
        self.right_ts = scaler.fit_transform(pivot_right.to_numpy(dtype=float))

        return pivot_left, pivot_right

    def run_clustering_experiments(self, side="left", methods=["kmeans", "dtw", "kshape", "fpca", "autoencoder"]):
        ts = self.left_ts if side == "left" else self.right_ts
        # ts shape from tslearn = (n_patients, n_visits, 1)

        # flatten tslearn tensor → (n_patients, n_visits)
        ts_flat = ts.reshape(ts.shape[0], ts.shape[1])
        results = {}

        for n_clusters in tqdm(self.n_clusters_list, desc=f"{side.upper()} clusters"):
            results[n_clusters] = {}

            for method in methods:
                logging.info(f"Running {method} clustering with {n_clusters} clusters on {side} pain")

                if method == "kmeans":
                    model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                    labels = model.fit_predict(ts_flat)

                elif method == "dtw":
                    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=self.random_state)
                    labels = model.fit_predict(ts)

                elif method == "kshape":
                    model = KShape(n_clusters=n_clusters, random_state=self.random_state)
                    labels = model.fit_predict(ts)

                elif method == "fpca":
                    # Functional PCA using standard PCA on flattened trajectories
                    X_flat = ts_flat
                    scaler_fpca = StandardScaler()
                    X_scaled = scaler_fpca.fit_transform(X_flat)
                    pca = PCA(n_components=min(5, ts.shape[1]))
                    X_pca = pca.fit_transform(X_scaled)
                    model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                    labels = model.fit_predict(X_pca)

                elif method == "autoencoder":
                    # Simple autoencoder for dimensionality reduction
                    X = ts_flat
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    class AE(nn.Module):
                        def __init__(self, input_dim, latent_dim=3):
                            super().__init__()
                            self.encoder = nn.Sequential(
                                nn.Linear(input_dim, 32),
                                nn.ReLU(),
                                nn.Linear(32, latent_dim)
                            )
                            self.decoder = nn.Sequential(
                                nn.Linear(latent_dim, 32),
                                nn.ReLU(),
                                nn.Linear(32, input_dim)
                            )
                        def forward(self, x):
                            z = self.encoder(x)
                            x_recon = self.decoder(z)
                            return x_recon, z
                    ae = AE(input_dim=X.shape[1], latent_dim=3).to(self.device)
                    optimizer = torch.optim.Adam(ae.parameters(), lr=0.01)
                    loss_fn = nn.MSELoss()

                    for epoch in range(200):
                        ae.train()
                        optimizer.zero_grad()
                        x_recon, z = ae(X_tensor)
                        loss = loss_fn(x_recon, X_tensor)
                        loss.backward()
                        optimizer.step()

                    ae.eval()
                    with torch.no_grad():
                        _, z = ae(X_tensor)
                        z_np = z.cpu().numpy()
                        model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                        labels = model.fit_predict(z_np)

                else:
                    continue

                results[n_clusters][method] = {"labels": labels, "model": model}

                pd.Series(labels, name=f"{side}_{method}_{n_clusters}").to_csv(
                    os.path.join(self.save_root, f"{side}_{method}_{n_clusters}_labels.csv"), index=False
                )
                self.plot_clusters(ts, labels, side, method, n_clusters)

        return results


    def plot_clusters(self, ts, labels, side, method, n_clusters):
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)

        for cluster in unique_labels:
            cluster_ts = ts[labels == cluster]
            for series in cluster_ts:
                plt.plot(series.ravel(), color='gray', alpha=0.05)

            centroid = cluster_ts.mean(axis=0)
            plt.plot(centroid.ravel(), label=f"cluster {cluster}", linewidth=2)

        plt.title(f"{side.upper()} pain trajectories - {method} ({n_clusters} clusters)")
        plt.xlabel("Visit Index")
        plt.ylabel("Standardized Pain Score")
        plt.legend()

        save_path = os.path.join(self.figure_root, f"{side}_{method}_{n_clusters}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved cluster figure: {save_path}")

    def merge_clusters_into_df(self, labels_left, labels_right, method="dtw", n_clusters=4):
        pivot_left, pivot_right = self.extract_trajectories()

        df_left = pivot_left.copy()
        df_left[f"left_cluster_{method}_{n_clusters}"] = labels_left

        df_right = pivot_right.copy()
        df_right[f"right_cluster_{method}_{n_clusters}"] = labels_right

        merged = self.df.merge(df_left[[f"left_cluster_{method}_{n_clusters}"]], on="ID", how="left")
        merged = merged.merge(df_right[[f"right_cluster_{method}_{n_clusters}"]], on="ID", how="left")

        output_path = "MDLPOPT/data/processed/all_visits_pain_trajectory_clusters.csv"
        merged.to_csv(output_path, index=False)
        logging.info(f"Saved merged dataframe with cluster labels: {output_path}")
        return merged

    # --------------------------------------------------
    # Find optimal cluster number using silhouette and elbow
    # --------------------------------------------------
    def evaluate_optimal_clusters(self, side="left", method="kmeans"):
        ts = self.left_ts if side == "left" else self.right_ts
        silhouette_scores = []
        inertias = []

        for n_clusters in self.n_clusters_list:
            if method == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                labels = model.fit_predict(ts.reshape(ts.shape[0], -1))
                inertia = model.inertia_
            elif method == "dtw":
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=self.random_state)
                labels = model.fit_predict(ts)
                inertia = None
            elif method == "kshape":
                model = KShape(n_clusters=n_clusters, random_state=self.random_state)
                labels = model.fit_predict(ts)
                inertia = None
            else:
                continue

            score = silhouette_score(ts.reshape(ts.shape[0], -1), labels)
            silhouette_scores.append(score)
            inertias.append(inertia)

        # Plot silhouette scores
        plt.figure(figsize=(8,4))
        plt.plot(self.n_clusters_list, silhouette_scores, marker='o')
        plt.title(f"Silhouette Scores - {side.upper()} - {method}")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.show()

        # Optionally, plot elbow for KMeans
        if method == "kmeans":
            plt.figure(figsize=(8,4))
            plt.plot(self.n_clusters_list, inertias, marker='o')
            plt.title(f"Elbow Plot (KMeans) - {side.upper()}")
            plt.xlabel("Number of clusters")
            plt.ylabel("Inertia")
            plt.grid(True)
            plt.show()

        logging.info("Evaluation complete. Use plots to select optimal number of clusters.")
        return silhouette_scores, inertias
