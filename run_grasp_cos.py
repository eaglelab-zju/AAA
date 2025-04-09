import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from the_utils import save_to_csv_files, set_device, set_seed


def intra_similarity(feature_dict, labels, feature_type, device):
    """
    类内平均相似度：每个类的所有类内节点两两算相似度，求整体均值
    """
    unique_labels = torch.unique(labels)
    total_sim = 0.0
    total_classes = 0

    for label in unique_labels:
        mask = labels == label
        class_features = [
            feature_dict[page_id][feature_type].to(device)
            for page_id, label_val in zip(feature_dict.keys(), labels)
            if label_val == label
        ]
        class_features = torch.stack(class_features).to(device)

        n = class_features.size(0)
        if n < 2:
            continue

        sim_matrix = F.cosine_similarity(
            class_features.unsqueeze(1), class_features.unsqueeze(0), dim=-1
        )
        triu_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        class_sim = sim_matrix[triu_mask].mean().item()

        total_sim += class_sim
        total_classes += 1

    return total_sim / total_classes if total_classes > 0 else 0.0


def cluster_similarity(feature_dict, sampled_page_ids, feature_type, device):
    """
    采样节点相似度：采样的节点两两算相似度，求均值
    """
    if len(sampled_page_ids) < 2:
        return 0.0

    sampled_features = [feature_dict[pid][feature_type].to(device) for pid in sampled_page_ids]
    sampled_features = torch.stack(sampled_features).to(device)

    n = len(sampled_page_ids)
    sim_matrix = F.cosine_similarity(
        sampled_features.unsqueeze(1), sampled_features.unsqueeze(0), dim=-1
    )
    triu_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    return sim_matrix[triu_mask].mean().item()


def split_graph_features(graph, page_ids, device):
    """
    将图的节点特征拆分为 a11y_emb、text_emb 和 image_emb，并与 page_id 对应
    """
    features = graph.ndata["feat"].to(device)

    a11y_emb = features[:, :131]
    text_emb = features[:, 131 : 131 + 768]
    image_emb = features[:, 131 + 768 :]

    feature_dict = {}
    for idx, page_id in enumerate(page_ids):
        feature_dict[page_id] = {
            "a11y_emb": a11y_emb[idx],
            "text_emb": text_emb[idx],
            "image_emb": image_emb[idx],
        }

    return feature_dict


def process_sampled_page_ids(df, column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: x.strip("[]").replace("'", "").split())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cos",
        description="cos",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = set_device(args.gpu_id)

    EMB_PATH = Path("./data")
    GRAPH_PATH = Path("./tmp/grasp/graphs")
    GRASP_PATH = EMB_PATH / "grasp"
    GRASP_CSV = Path("./results/grasp.csv")
    SDC_PATH = EMB_PATH / "sdc"
    SDC_CSV = Path("./results/sdc.csv")
    DOM_PATH = Path("../UIST_DOMData")

    grasp_df = pd.read_csv(GRASP_CSV)
    grasp_df = process_sampled_page_ids(grasp_df, ["sampled_page_ids"])

    results_a11y = []
    results_text = []
    results_image = []

    for folder in GRASP_PATH.iterdir():
        folder_name = folder.name
        folder_graph_path = GRAPH_PATH / folder_name
        feature = next(
            (f for f in os.listdir(folder_graph_path) if folder_graph_path.joinpath(f).is_file()),
            None,
        )
        feature_path = folder_graph_path / feature
        graph, page_ids = torch.load(feature_path, map_location=device)
        feature_dict = split_graph_features(graph, page_ids, device)

        # 计算grasp的类内平均相似度和采样节点相似度
        folder_path = folder
        emb_file = next(folder_path.iterdir())
        z_grasp, labels, centers = torch.load(emb_file, map_location=device)
        emb_file_name = emb_file.name

        grasp_row = grasp_df[grasp_df["url"] == emb_file_name].iloc[0]
        sampled_page_ids = grasp_row["sampled_page_ids"]

        # grasp的类内平均相似度
        labels = torch.tensor(labels).to(device)
        intra_sim_grasp_a11y = intra_similarity(feature_dict, labels, "a11y_emb", device)
        intra_sim_grasp_text = intra_similarity(feature_dict, labels, "text_emb", device)
        intra_sim_grasp_image = intra_similarity(feature_dict, labels, "image_emb", device)

        # grasp的采样节点相似度
        cluster_sim_grasp_a11y = cluster_similarity(
            feature_dict, sampled_page_ids, "a11y_emb", device
        )
        cluster_sim_grasp_text = cluster_similarity(
            feature_dict, sampled_page_ids, "text_emb", device
        )
        cluster_sim_grasp_image = cluster_similarity(
            feature_dict, sampled_page_ids, "image_emb", device
        )

        save_to_csv_files(
            results={
                "model": "grasp",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_a11y,
                "cluster_sim": cluster_sim_grasp_a11y,
            },
            csv_name="a11y_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": "grasp",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_text,
                "cluster_sim": cluster_sim_grasp_text,
            },
            csv_name="text_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": "grasp",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_image,
                "cluster_sim": cluster_sim_grasp_image,
            },
            csv_name="image_emb.csv",
        )

print("Done")
