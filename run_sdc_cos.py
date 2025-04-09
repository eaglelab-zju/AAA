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
        df[column_name] = df[column_name].apply(
            lambda x: x.strip("[]").replace("'", "").split()
        )
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

    sdc_df = pd.read_csv(SDC_CSV)
    sdc_df = process_sampled_page_ids(
        sdc_df,
        [
            "tags_kmeans_c_sampled_page_ids",
            "tags_tsne_kmeans_c_sampled_page_ids",
            "structure_kmeans_c_sampled_page_ids",
            "structure_tsne_kmeans_c_sampled_page_ids",
            "content_kmeans_c_sampled_page_ids",
            "content_tsne_kmeans_c_sampled_page_ids",
            "struc_cont_kmeans_c_sampled_page_ids",
            "struc_cont_tsne_kmeans_c_sampled_page_ids",
            "tree_kmeans_c_sampled_page_ids",
            "tree_tsne_kmeans_c_sampled_page_ids",
        ],
    )

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

        folder_path = folder
        emb_file = next(folder_path.iterdir())
        z_grasp, labels, centers = torch.load(emb_file, map_location=device)
        emb_file_name = emb_file.name

        # 计算sdc的类内平均相似度和采样节点相似度
        sdc_folder_path = SDC_PATH / emb_file_name
        for sdc_type in ["content", "struc_cont", "structure", "tags", "tree"]:
            sdc_file = sdc_folder_path / sdc_type
            data = torch.load(sdc_file, map_location=device)
            dense_matrix, tsne_matrix, dense_labels, dense_c, tsne_labels, tsne_c = data

            # 计算SDC类内相似度
            dense_labels = dense_labels.clone().detach().to(device)
            tsne_labels = tsne_labels.clone().detach().to(device)
            intra_sim_dense_a11y = intra_similarity(feature_dict, dense_labels, "a11y_emb", device)
            intra_sim_dense_text = intra_similarity(feature_dict, dense_labels, "text_emb", device)
            intra_sim_dense_image = intra_similarity(
                feature_dict, dense_labels, "image_emb", device
            )
            intra_sim_tsne_a11y = intra_similarity(feature_dict, tsne_labels, "a11y_emb", device)
            intra_sim_tsne_text = intra_similarity(feature_dict, tsne_labels, "text_emb", device)
            intra_sim_tsne_image = intra_similarity(feature_dict, tsne_labels, "image_emb", device)

            # 计算SDC采样节点相似度
            sdc_row = sdc_df[sdc_df["url"] == emb_file_name].iloc[0]
            dense_sampled_page_ids = sdc_row[f"{sdc_type}_kmeans_c_sampled_page_ids"]
            tsne_sampled_page_ids = sdc_row[f"{sdc_type}_tsne_kmeans_c_sampled_page_ids"]

            cluster_sim_dense_a11y = cluster_similarity(
                feature_dict, dense_sampled_page_ids, "a11y_emb", device
            )
            cluster_sim_dense_text = cluster_similarity(
                feature_dict, dense_sampled_page_ids, "text_emb", device
            )
            cluster_sim_dense_image = cluster_similarity(
                feature_dict, dense_sampled_page_ids, "image_emb", device
            )
            cluster_sim_tsne_a11y = cluster_similarity(
                feature_dict, tsne_sampled_page_ids, "a11y_emb", device
            )
            cluster_sim_tsne_text = cluster_similarity(
                feature_dict, tsne_sampled_page_ids, "text_emb", device
            )
            cluster_sim_tsne_image = cluster_similarity(
                feature_dict, tsne_sampled_page_ids, "image_emb", device
            )

            save_to_csv_files(
                results={
                    "model": f"sdc_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_dense_a11y,
                    "cluster_sim": cluster_sim_dense_a11y,
                },
                csv_name="sdc/a11y_emb.csv",
            )
            save_to_csv_files(
                results={
                    "model": f"sdc_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_dense_text,
                    "cluster_sim": cluster_sim_dense_text,
                },
                csv_name="sdc/text_emb.csv",
            )
            save_to_csv_files(
                results={
                    "model": f"sdc_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_dense_image,
                    "cluster_sim": cluster_sim_dense_image,
                },
                csv_name="sdc/image_emb.csv",
            )
            save_to_csv_files(
                results={
                    "model": f"sdc_tsne_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_tsne_a11y,
                    "cluster_sim": cluster_sim_tsne_a11y,
                },
                csv_name="sdc/a11y_emb.csv",
            )
            save_to_csv_files(
                results={
                    "model": f"sdc_tsne_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_tsne_text,
                    "cluster_sim": cluster_sim_tsne_text,
                },
                csv_name="sdc/text_emb.csv",
            )
            save_to_csv_files(
                results={
                    "model": f"sdc_tsne_{sdc_type}",
                    "url": emb_file_name,
                    "intra_sim": intra_sim_tsne_image,
                    "cluster_sim": cluster_sim_tsne_image,
                },
                csv_name="sdc/image_emb.csv",
            )
print("Done")