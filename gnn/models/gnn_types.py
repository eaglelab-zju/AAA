import os
import sys
from pathlib import Path
from typing import List

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../../baseline/baselines")
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE
from OrderedGNN import OrderedGNN
from SGFormer import SGFormer
from SIGN import SIGN, preprocess


# ====================== GCN Wrapper ==========================
class GCNWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int],
        n_gnn_layers: int,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _GCNArgs:
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            layers = n_gnn_layers
            nhidden = hidden_units[-1] if hidden_units else in_feats
            dropout = 0.5
            patience = 100

        self.gcn = GCN(in_features=in_feats, class_num=n_clusters, device=device, args=_GCNArgs())

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, hidden_units[-1]))
        nn.init.orthogonal_(self.cluster_layer.data)

    def forward(self, graph, features):
        graph = dgl.add_self_loop(graph)
        return self.gcn(graph, features)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                return self.best_model(graph, features)
            else:
                return self.forward(graph, features)


# ====================== GAT Wrapper ==========================
class GATWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int],
        n_gnn_layers: int,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _GATArgs:
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            layers = n_gnn_layers
            nhidden = hidden_units[-1] if hidden_units else in_feats
            dropout = 0.5
            patience = 100

        self.gat = GAT(in_features=in_feats, class_num=n_clusters, device=device, args=_GATArgs())

        self.cluster_layer = nn.Parameter(
            torch.Tensor(n_clusters, hidden_units[-1] if hidden_units else in_feats)
        )
        nn.init.orthogonal_(self.cluster_layer.data)

    def forward(self, graph, features):
        graph = dgl.add_self_loop(graph)
        return self.gat(graph, features)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                return self.best_model(graph, features)
            else:
                return self.forward(graph, features)


# ====================== GraphSAGE Wrapper =====================
class graphSAGEWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int],
        n_gnn_layers: int,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _SAGEArgs:
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            layers = n_gnn_layers
            nhidden = hidden_units[-1] if hidden_units else in_feats
            dropout = 0.5
            patience = 100

        self.graphsage = GraphSAGE(
            in_features=in_feats, class_num=n_clusters, device=device, args=_SAGEArgs()
        )

        self.cluster_layer = nn.Parameter(
            torch.Tensor(n_clusters, hidden_units[-1] if hidden_units else in_feats)
        )
        nn.init.orthogonal_(self.cluster_layer.data)

    def forward(self, graph, features):
        graph = dgl.add_self_loop(graph)
        return self.graphsage(graph, features)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                return self.best_model(graph, features)
            else:
                return self.forward(graph, features)


# ====================== OrderedGNN Wrapper =====================
class OrderedGNNWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int],
        n_gnn_layers: int,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _OrderedGNNAgs:
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            hidden_channel = hidden_units[-1] if hidden_units else in_feats
            num_layers_input = kwargs.get("num_layers_input", 1)
            num_layers = n_gnn_layers
            dropout = kwargs.get("dropout", 0.5)
            dropout2 = kwargs.get("dropout2", "None")
            chunk_size = kwargs.get("chunk_size", 1)
            add_self_loops = kwargs.get("add_self_loops", True)
            simple_gating = kwargs.get("simple_gating", False)
            tm = kwargs.get("tm", True)
            diff_or = kwargs.get("diff_or", True)
            patience = kwargs.get("patience", 100)
            global_gating = kwargs.get("global_gating", False)

        self.ordered_gnn = OrderedGNN(
            in_features=in_feats,
            class_num=n_clusters,
            device=device,
            args=_OrderedGNNAgs(),
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, hidden_units[-1]))
        nn.init.orthogonal_(self.cluster_layer.data)
        self.device = device

    def forward(self, graph, features, return_Z=False):
        adj = graph.adj_external(scipy_fmt="csr")
        edge_index = torch.tensor(np.array(adj.nonzero()), device=self.device, dtype=torch.long)
        return self.ordered_gnn(features, edge_index, return_Z=return_Z)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                Z, _ = self.best_model.forward(graph, features, return_Z=True)
                return Z
            else:
                Z, _ = self.forward(graph, features, return_Z=True)
                return Z


# ====================== SGFormer Wrapper =====================
class SGFormerWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int] = [512, 256, 20],
        n_gnn_layers: int = 1,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _SGFormerArgs:
            nhidden = hidden_units[-1] if hidden_units else in_feats
            num_layers = n_gnn_layers
            num_heads = kwargs.get("num_heads", 1)
            alpha = kwargs.get("alpha", 0.5)
            dropout = kwargs.get("dropout", 0.5)
            use_bn = kwargs.get("use_bn", True)
            use_residual = kwargs.get("use_residual", True)
            use_weight = kwargs.get("use_weight", True)
            use_graph = kwargs.get("use_graph", True)
            use_act = kwargs.get("use_act", False)
            graph_weight = kwargs.get("graph_weight", 0.5)
            gnn = kwargs.get("gnn", "gcn")
            aggregate = kwargs.get("aggregate", "add")
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            patience = kwargs.get("patience", 100)

        self.sgformer = SGFormer(
            in_features=in_feats,
            class_num=n_clusters,
            device=device,
            args=_SGFormerArgs(),
        )
        self.cluster_layer = nn.Parameter(
            torch.Tensor(n_clusters, hidden_units[-1] if hidden_units else in_feats)
        )
        nn.init.orthogonal_(self.cluster_layer.data)
        self.device = device
        if self.sgformer.args.aggregate == "concat":
            self.projection = nn.Linear(hidden_units[-1] * 2, hidden_units[-1])

    def forward(self, graph, features):
        edge_index = torch.stack(graph.edges()).to(self.device)
        return self.sgformer(features, edge_index)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                model = self.best_model
            else:
                model = self
            edge_index = torch.stack(graph.edges()).to(self.device)
            x1 = model.sgformer.trans_conv(features, edge_index)
            if model.sgformer.use_graph:
                x2 = model.sgformer.gnn(features, edge_index)
                if model.sgformer.aggregate == "add":
                    x = x1 + x2
                elif model.sgformer.aggregate == "concat":
                    x = torch.cat([x1, x2], dim=1)
                    x = self.projection(x)
            else:
                x = x1
            return x


# ====================== SIGN Wrapper =====================
class SIGNWrapper(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        hidden_units: List[int] = [512, 256, 20],
        n_gnn_layers: int = 1,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()

        class _SIGNArgs:
            nhidden = hidden_units[-1] if hidden_units else in_feats
            n_hops = kwargs.get("n_hops", 3)
            n_layers = kwargs.get("n_layers", 3)
            dropout = kwargs.get("dropout", 0.5)
            lr = kwargs.get("lr", 0.001)
            l2_coef = kwargs.get("l2_coef", 0.0)
            epochs = kwargs.get("epochs", 50)
            patience = kwargs.get("patience", 100)

        self.sign = SIGN(
            in_features=in_feats,
            class_num=n_clusters,
            device=device,
            args=_SIGNArgs(),
        )
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, hidden_units[-1]))
        nn.init.orthogonal_(self.cluster_layer.data)
        self.device = device
        self.hops = _SIGNArgs.n_hops
        self.projection = nn.Linear(80, hidden_units[-1])

    def forward(self, graph, features):
        graph = graph.remove_self_loop().add_self_loop()
        preprocessed_feats = preprocess(graph, features, self.hops)
        return self.sign(preprocessed_feats)

    def get_embedding(self, graph, features, best=True):
        with torch.no_grad():
            if best and hasattr(self, "best_model"):
                model = self.best_model
            else:
                model = self
            graph = graph.remove_self_loop().add_self_loop()
            preprocessed_feats = preprocess(graph, features, self.hops)
            hidden = [ff(feat) for feat, ff in zip(preprocessed_feats, model.sign.inception_ffs)]
            z = torch.cat(hidden, dim=-1)
            z = self.projection(z)
            return z


GNN_TYPE_REGISTRY = {
    "gcn": GCNWrapper,
    "gat": GATWrapper,
    "graphSAGE": graphSAGEWrapper,
    "OrderedGNN": OrderedGNNWrapper,
    "SGFormer": SGFormerWrapper,
    "sign": SIGNWrapper,
}
