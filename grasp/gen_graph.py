"""generate graphs.
"""

# pylint: disable=too-many-locals,broad-exception-caught,invalid-name
import csv
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import shuffle
from typing import Dict, List

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from the_utils import make_parent_dirs
from torchvision.models import ViT_B_16_Weights, vit_b_16
from transformers import BertModel, BertTokenizer

from .preprocess import gen_screenshot_embedding, gen_text_embedding


def load_bert_model(bert_name="bert-base-multilingual-cased"):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    model = BertModel.from_pretrained(bert_name)
    return tokenizer, model


def load_vit_model():
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    return vit


def load_adj_matrix(file_path):
    return sp.csr_matrix(np.loadtxt(file_path))


def load_custom_data(adj_matrix, features, labels, device):
    graph = dgl.from_scipy(adj_matrix).to(device)

    graph.ndata["feat"] = features = torch.stack(
        [features[page_id].squeeze() for page_id in sorted(features.keys())]
    ).to(device)
    if labels is None:
        labels = torch.zeros(graph.num_nodes(), dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long)

    graph.ndata["label"] = labels.to(device)

    return graph


def get_rule_ids(path="./grasp/resource"):
    with open(f"{path}/rule_ids.txt", "r", encoding="utf-8") as file:
        rule_ids = [line.strip() for line in file]
    return rule_ids


def process_page(
    page_id,
    subfolder: Path,
    base_path: Path,
    hctask_id,
    rule_ids: List,
    tokenizer,
    bert,
    vit,
    device,
):
    t = time.time()
    a11y_emb = torch.zeros((1, 131), device=device)
    text_emb = torch.zeros((1, 768), device=device)
    image_emb = torch.zeros((1, 1000), device=device)

    html_path = subfolder.joinpath(f"{page_id}")
    if html_path.exists():
        try:
            text_emb = gen_text_embedding(html_path, tokenizer, bert, device, text_emb, page_id)
        except Exception:
            traceback.print_exc()

    jpg_path = subfolder.joinpath(f"{page_id}.jpg")
    if jpg_path.exists():
        try:
            image_emb = gen_screenshot_embedding(jpg_path, vit, device)
        except Exception:
            traceback.print_exc()
    else:
        print(f"page {page_id} has no screenshot.", flush=True)

    axe_path = base_path.joinpath(f"UIST_axeData/axe_res/{hctask_id}/{page_id}.json")
    if axe_path.exists():
        with open(axe_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        ids = [
            item["id"]
            for key in ["incomplete", "violations", "inapplicable"]
            for item in data.get(key, [])
        ]
        id_counts = Counter(ids)
        a11y_emb = F.normalize(
            torch.tensor(
                [id_counts.get(rule_id, 0) for rule_id in rule_ids],
                dtype=torch.float,
                device=device,
            ).unsqueeze(dim=0)
        )
    else:
        print(f"page {page_id} has no axe result.", flush=True)

    print(f"{page_id} done, Time:{time.time()-t:.2f}", flush=True)
    return page_id, torch.cat((a11y_emb, text_emb, image_emb), dim=-1).to(torch.device("cpu"))


def gen_graphs(
    graph_path: Path,
    subfolder: Path,
    subfolder_name: str,
    base_path: Path,
    rule_ids: list,
    tokenizer,
    bert,
    vit,
    device=torch.device("cpu"),
) -> dgl.DGLGraph:

    if graph_path.exists():
        print(f"Load graph from cache {graph_path}")
        return torch.load(graph_path, map_location=device)

    make_parent_dirs(graph_path)
    if not subfolder_name:
        print(f"{subfolder} has no DOM. Skip.")
        return None, None

    subfolder_name = str(subfolder_name)
    hctask_id = subfolder_name.split("-", maxsplit=1)[0]

    adj_matrix_path = base_path.joinpath(f"UIST_GraphData/{hctask_id}/adj_matrix.txt")

    fs = sorted(
        [f for f in os.listdir(subfolder) if not f.lower().endswith((".jpg", ".logs", ".log"))]
    )
    shuffle(fs)

    SEP = "=" * 6
    print(f"\n\n{SEP}{subfolder_name} {len(fs)} gen_graphs START{SEP}")
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count())) as executor:
        results = executor.map(
            process_page,
            fs,
            [subfolder] * len(fs),
            [base_path] * len(fs),
            [hctask_id] * len(fs),
            [rule_ids] * len(fs),
            [tokenizer] * len(fs),
            [bert] * len(fs),
            [vit] * len(fs),
            [device] * len(fs),
            chunksize=4,
        )

    page_features = {}
    for page_id, feature in results:
        page_features[page_id] = feature
    page_ids = sorted(page_features.keys())

    graph = load_custom_data(
        load_adj_matrix(adj_matrix_path), page_features, labels=None, device=device
    )
    torch.save((graph, page_ids), f=graph_path)
    print(f"{SEP}{subfolder_name} END{SEP}\n\n")

    return graph, page_ids
