import copy
import os
import random
import sys

import dgl
import torch
from models.gnn import GNN

sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../hole")
from utils.utils import check_modelfile_exists, load_model, save_model


def train_gnn(folder, graph: dgl.DGLGraph, features, n_clusters, device, model_type):
    final_params = {
        "dim": 500,
        "n_gnn_layers": 1,
        "n_lin_layers": 1,
        "lr": 0.001,
        "pre_epochs": 150,
        "epochs": 50,
        "udp": 10,
        "inner_act": torch.nn.Identity(),
        "add_edge_ratio": 0.5,
        "node_ratio": 1,
        "del_edge_ratio": 0.01,
        "regularization": 1,
        "runs": 1,
        "device": device,
    }

    warmup_filename = f"{model_type}/warmup_{model_type}"
    adj_sum_raw = graph.adj_external(scipy_fmt="csr").sum()

    if not check_modelfile_exists(warmup_filename):
        print(f"Initializing {model_type} warmup...")
        model = GNN(
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            model_type=model_type,
            warmup_filename=warmup_filename,
            **final_params,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = graph.to(device)
        features = features.to(device)
        model = model.to(device)
        model.fit(graph, device, gsl_epochs=0)

    seed_list = [random.randint(0, 999999) for _ in range(final_params["runs"])]
    for run_id in range(final_params["runs"]):
        model = GNN(
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            model_type=model_type,
            warmup_filename=warmup_filename,
            **final_params,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = graph.to(device)
        features = features.to(device)
        model = model.to(device)

        model.fit(
            graph,
            device,
            add_edge_ratio=final_params["add_edge_ratio"],
            node_ratio=final_params["node_ratio"],
            del_edge_ratio=final_params["del_edge_ratio"],
        )

    with torch.no_grad():
        z_detached = model.get_embedding(graph, features)
        Q = model.get_Q(z=z_detached)
        q = Q.detach().data.cpu().numpy().argmax(1)
        model.get_cluster_center(z=z_detached)
        c = model.cluster_layer

    return z_detached, q, c
