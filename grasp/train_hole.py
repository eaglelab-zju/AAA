"""Train HoLe."""

# pylint:disable=line-too-long,wrong-import-position,invalid-name,too-many-locals,unused-variable
import os
import random
import sys

import dgl
import torch

sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../hole")
from hole.models import HoLe
from hole.utils import check_modelfile_exists, get_str_time


def train_hole(folder, graph: dgl.DGLGraph, features, n_clusters, device, type):
    """train HoLe."""
    final_params = {}
    dim = 500
    n_lin_layers = 1
    dump = True
    lr = 0.001
    n_gnn_layer = 1
    pre_epoch = 150
    epochs = 50
    inner_act = torch.nn.Identity()
    udp = 10
    node_ratio = 1
    add_edge_ratio = 0.5
    del_edge_ratio = 0.01
    gsl_epochs = 5
    regularization = 1
    runs = 1

    final_params["dim"] = dim
    final_params["n_gnn_layers"] = n_gnn_layer
    final_params["n_lin_layers"] = n_lin_layers
    final_params["lr"] = lr
    final_params["pre_epochs"] = pre_epoch
    final_params["epochs"] = epochs
    final_params["udp"] = udp
    final_params["inner_act"] = inner_act
    final_params["add_edge_ratio"] = add_edge_ratio
    final_params["node_ratio"] = node_ratio
    final_params["del_edge_ratio"] = del_edge_ratio
    final_params["gsl_epochs"] = gsl_epochs

    time_name = get_str_time()
    if type == "ignn":
        save_file = f"results/hole/ignn/hole_custom_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}_{folder}.csv"
        warmup_filename = f"ignn/hole_custom_run_gnn_{n_gnn_layer}"
        ignn_path = "ignn/"
    else:
        save_file = f"results/hole/hole_custom_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}_{folder}.csv"
        warmup_filename = f"hole_custom_run_gnn_{n_gnn_layer}"
        ignn_path = ""

    adj_sum_raw = graph.adj_external(scipy_fmt="csr").sum()

    if not check_modelfile_exists(warmup_filename):
        print("warmup first")
        model = HoLe(
            hidden_units=[dim],
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            n_gnn_layers=n_gnn_layer,
            n_lin_layers=n_lin_layers,
            lr=lr,
            n_pretrain_epochs=pre_epoch,
            n_epochs=epochs,
            norm="sym",
            renorm=True,
            tb_filename=f"{ignn_path}custom_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio}_{del_edge_ratio}_pre_ep{pre_epoch}_ep{epochs}_dim{dim}_{random.randint(0, 999999)}",
            warmup_filename=warmup_filename,
            inner_act=inner_act,
            udp=udp,
            regularization=regularization,
            type=type,
        )

        model.fit(
            graph=graph,
            device=device,
            add_edge_ratio=add_edge_ratio,
            node_ratio=node_ratio,
            del_edge_ratio=del_edge_ratio,
            gsl_epochs=0,
            adj_sum_raw=adj_sum_raw,
            load=False,
            dump=dump,
        )

    seed_list = [random.randint(0, 999999) for _ in range(runs)]
    for run_id in range(runs):
        final_params["run_id"] = run_id
        seed = seed_list[run_id]
        final_params["seed"] = seed

        model = HoLe(
            hidden_units=[dim],
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            n_gnn_layers=n_gnn_layer,
            n_lin_layers=n_lin_layers,
            lr=lr,
            n_pretrain_epochs=pre_epoch,
            n_epochs=epochs,
            norm="sym",
            renorm=True,
            tb_filename=f"{ignn_path}custom_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs}_dim{dim}_{random.randint(0, 999999)}",
            warmup_filename=warmup_filename,
            inner_act=inner_act,
            udp=udp,
            reset=False,
            regularization=regularization,
            seed=seed,
            type=type,
        )

        model.fit(
            graph=graph,
            device=device,
            add_edge_ratio=add_edge_ratio,
            node_ratio=node_ratio,
            del_edge_ratio=del_edge_ratio,
            gsl_epochs=gsl_epochs,
            adj_sum_raw=adj_sum_raw,
            load=False,
            dump=dump,
        )

        with torch.no_grad():
            z_detached = model.get_embedding()
            Q = model.get_Q(z=z_detached)
            q = Q.detach().data.cpu().numpy().argmax(1)
            model.get_cluster_center(z=z_detached)
            c = model.cluster_layer

    return z_detached, q, c
