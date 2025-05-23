import copy
import os
import sys
import time
from typing import Callable, List

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .gnn_types import GNN_TYPE_REGISTRY

sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../../hole")
from modules import InnerProductDecoder, SampleDecoder
from utils.utils import check_modelfile_exists, load_model, save_model


class GNN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_clusters: int,
        model_type: str = "gcn",
        hidden_units: List[int] = [512, 256, 20],
        n_gnn_layers: int = 1,
        dropout: float = 0.5,
        n_pretrain_epochs: int = 100,
        n_epochs: int = 200,
        lr: float = 0.001,
        l2_coef: float = 0.0,
        regularization: float = 0.1,
        inner_act: Callable = lambda x: x,
        udp: int = 10,
        warmup_filename: str = "gcn_warmup",
        **kwargs,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_pretrain_epochs = n_pretrain_epochs
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_coef = l2_coef
        self.regularization = regularization
        self.udp = udp
        self.warmup_filename = warmup_filename

        assert model_type in GNN_TYPE_REGISTRY, f"Unsupported model type: {model_type}"
        self.model = GNN_TYPE_REGISTRY[model_type](
            in_feats=in_feats,
            hidden_units=hidden_units,
            dropout=dropout,
            n_clusters=n_clusters,
            n_gnn_layers=n_gnn_layers,
            **kwargs,
        )

        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, hidden_units[-1]))
        torch.nn.init.orthogonal_(self.cluster_layer.data)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sample_decoder = SampleDecoder(act=lambda x: x)
        self.inner_product_decoder = InnerProductDecoder(act=inner_act)

        self.best_model = None

    def forward(self, g, features):
        return self.model(g, features)

    def get_embedding(self, g, features):
        return self.model.get_embedding(g, features)

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def _target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_cluster_center(self, z):
        try:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            kmeans.fit(z.detach().cpu().numpy())
            self.cluster_layer.data = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=z.device
            )
        except Exception as e:
            print(e)

    def _pretrain(self, g, features, adj_label, pos_weight, norm_weight):
        if self.best_model is None:
            self.best_model = copy.deepcopy(self.state_dict())
        best_loss = float("inf")
        for epoch in range(self.n_pretrain_epochs):
            t = time.time()
            self.train()
            self.optimizer.zero_grad()

            z = self(g, features)

            adj_recon = self.inner_product_decoder(z)
            recon_loss = (
                F.binary_cross_entropy_with_logits(
                    adj_recon.view(-1), adj_label.view(-1), pos_weight=pos_weight
                )
                * norm_weight
            )

            fd_loss = self._feature_decorrelation_loss(z)

            total_loss = recon_loss + self.regularization * fd_loss
            total_loss.backward()
            self.optimizer.step()

            if total_loss < best_loss:
                best_loss = total_loss
                self.best_model = copy.deepcopy(self.state_dict())

            print(
                f"Pretrain Epoch {epoch}: Loss={total_loss.item():.4f},"
                f"time={time.time() - t:.5f}"
            )

    def _train(self, g, features, adj_label, pos_weight, norm_weight):
        self.load_state_dict(self.best_model)
        with torch.no_grad():
            z = self.get_embedding(g, features)
            self.get_cluster_center(z)

        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            t = time.time()

            z = self(g, features)

            adj_recon = self.inner_product_decoder(z)
            recon_loss = (
                F.binary_cross_entropy_with_logits(
                    adj_recon.view(-1), adj_label.view(-1), pos_weight=pos_weight
                )
                * norm_weight
            )

            q = self.get_Q(z)
            p = self._target_distribution(q.detach())
            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")

            total_loss = recon_loss + kl_loss
            total_loss.backward()
            self.optimizer.step()

            print(
                f"Cluster Epoch: {epoch}, embeds_loss={total_loss:.5f},"
                f"kl_loss={kl_loss.item()},"
                f"time={time.time() - t:.5f}"
            )

    def fit(self, graph, device, load=False, dump=False, **kwargs):
        self.to(device)

        features = graph.ndata["feat"].to(device)
        adj = graph.adj_external(scipy_fmt="csr")
        adj_label = torch.FloatTensor(adj.toarray()).to(device)

        pos_weight = torch.tensor([(adj.shape[0] ** 2 - adj.sum()) / adj.sum()]).to(device)
        norm_weight = adj.shape[0] ** 2 / ((adj.shape[0] ** 2 - adj.sum()) * 2)

        if load and check_modelfile_exists(self.warmup_filename):
            from utils.utils import load_model

            self, self.optimizer, _, _ = load_model(
                self.warmup_filename,
                self,
                self.optimizer,
                self.device,
            )
            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain(graph, features, adj_label, pos_weight, norm_weight)
            if dump:
                from utils.utils import save_model

                save_model(
                    self.warmup_filename,
                    self,
                    self.optimizer,
                    None,
                    None,
                )
                print(f"dump to {self.warmup_filename}")

        self._train(graph, features, adj_label, pos_weight, norm_weight)

    @staticmethod
    def _feature_decorrelation_loss(z):
        norm_z = z / torch.norm(z, dim=0, keepdim=True)
        corr_matrix = torch.mm(norm_z.t(), norm_z)
        identity = torch.eye(corr_matrix.size(0), device=z.device)
        return F.mse_loss(corr_matrix, identity)
