import json
import os
import argparse
import matplotlib.pyplot as plt
import sys
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torch
from pathlib import Path

from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
main_root = os.getcwd()
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(main_root)
from src.utils import load_json_file, import_class


def cluster_embeddings_dbscan(embeddings, eps=0.5, min_samples=5, metric="euclidean"):
    """
    Normalize embeddings and run DBSCAN.
    """
    embeddings_norm = normalize(embeddings)

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )

    labels = dbscan.fit_predict(embeddings_norm)

    core_sample_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(dbscan, "core_sample_indices_"):
        core_sample_mask[dbscan.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print("DBSCAN finished")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    return embeddings_norm, labels, core_sample_mask


def compute_cluster_distance_score(embeddings_norm, labels):
    """
    DBSCAN does not provide probabilities.
    This function computes a simple diagnostic score:
    distance to the mean vector of the assigned cluster.

    For noise points (-1), score is NaN.

    Lower distance = more typical point for that cluster.
    """
    scores = np.full(len(labels), np.nan, dtype=float)

    unique_clusters = [c for c in np.unique(labels) if c != -1]

    for c in unique_clusters:
        idx = labels == c
        cluster_points = embeddings_norm[idx]

        if len(cluster_points) == 0:
            continue

        cluster_center = cluster_points.mean(axis=0, keepdims=True)
        dists = pairwise_distances(cluster_points, cluster_center, metric="euclidean").flatten()

        scores[idx] = dists

    return scores

################################ CONFIGURATION OF SCRIPTS #########################
parser = argparse.ArgumentParser()
parser.add_argument('--script_config', type=str, default='ResNet18_TrainingConfig_v0')
args = parser.parse_args()
# Read the json 

############################### EXTRACT THE CONFIGURATIONS ########################
script_config = args.script_config
script_config = load_json_file(Path(root_path, "fit_model_embeddings", "configs", f"{script_config}.json" ))
evaluation_label = script_config['general_parameters']["label"]
database_path =  Path(main_root) / Path(*script_config['general_parameters']["database_path"])
results_path = Path(main_root) / Path(*script_config['general_parameters']["path_to_save_results"])
os.makedirs(results_path,exist_ok=True)

############################### INITIALIZE THE TRAINING DATASET ####################
dataset_config = script_config["dataset"]
full_dataset = import_class(dataset_config['class_name'], 
                                dataset_config['module_name'])(**dataset_config['params'])
full_dataset.load_data(database_path, sub_folder = None)

############################## INITIALIZE THE MODEL TO FIT ##########################
model_config = script_config["model"]
model =  import_class(model_config['class_name'], 
                                model_config['module_name'])(**model_config['params'])

gmm = GaussianMixture(n_components=2,      # Las 2 clases que buscas
                    covariance_type='full', # Permite elipses complejas
                    n_init=10,           # Ejecuta 10 veces para encontrar el mejor fit
                    max_iter=300,        # Suficientes iteraciones para converger
                    random_state=42      # Para que los resultados sean reproducibles
                )

############################# SPLIT THE DATA ########################################
splitter_data = script_config["splitter_data"]
train_size = int(splitter_data['ptrain'] * len(full_dataset))
val_size = len(full_dataset) - train_size
gen = torch.Generator().manual_seed(splitter_data['seed'])
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                            generator=gen) 

X_train = []
for i in range(train_size):
    segmented_objects, metadata = train_dataset[i]
    X_embedding = model.evaluate_embeddings(segmented_objects)
    X_train.append(X_embedding)
X_train = torch.cat(X_train, dim=0)
############################# FIT THE MODEL #########################################
# X_pca = model._fit_PCA_pipeline(X_train)
# gmm.fit(X_pca)

# cov_matrix = np.cov(X_pca.T) + np.eye(X_pca.shape[1]) * 1e-6
# VI = np.linalg.inv(cov_matrix)

############################
datasets_to_process = [('train', train_dataset), ('val', val_dataset)]
results_objects = [] # DataFrame 1: Detalle por pez
results_list = []  # DataFrame 2: Resumen por frame
for subset_name, current_dataset in datasets_to_process:
    for i in tqdm(range(len(current_dataset)), desc=f"Procesando {subset_name}"):
        crops, metadata = current_dataset[i]
        if crops.shape[0] == 0: continue

        # Extraer embeddings y transformar con PCA
        
        X_embedding = model.evaluate_embeddings(crops).cpu().numpy()
        n_objetos = crops.shape[0]

        # DBSCAN Euclidean (Local)
        embeddings_norm, labels, core_sample_mask = cluster_embeddings_dbscan(
            X_embedding,
            eps=0.5,
            min_samples=5,
            metric="euclidean"
        )

        distance_score = compute_cluster_distance_score(embeddings_norm, labels)

        

        # --- CONSTRUCCIÓN DEL DATAFRAME (Formato Largo) ---
        for idx in range(n_objetos):
            base_info = {
                "frame_id": metadata[idx]['frame'],
                "video": metadata[idx]['video'],
                "bbox": metadata[idx]['bbox_yolo'],
                "subset": subset_name
            }

            # Registro GMM
            results_list.append({
                **base_info,
                "method": "dbscam_euclidean",
                "cluster_label": int(labels[idx]),
                "is_noise": bool(labels[idx] == -1),
                "is_core_sample": bool(core_sample_mask[idx]),
                "distance_to_cluster_mean": None if np.isnan(distance_score[idx]) else float(distance_score[idx])}           
                                )

# 2. Generar DataFrame de Objetos
df_results = pd.DataFrame(results_list)
df_results.to_csv(results_path / "clustering_results.csv", index=False)