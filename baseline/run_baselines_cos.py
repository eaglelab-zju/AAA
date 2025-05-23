import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from the_utils import save_to_csv_files, set_device, set_seed


def intra_similarity(feature_dict, labels, feature_type, device):
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


def calculate_averages_and_save_conclusion(model_type):
    result_files = [
        f"results/{model_type}/a11y_emb.csv",
        f"results/{model_type}/text_emb.csv",
        f"results/{model_type}/image_emb.csv",
    ]

    print(f"尝试读取以下文件：{result_files}")
    results = []

    for file_path in result_files:
        if os.path.exists(file_path):
            print(f"文件存在：{file_path}")
            df = pd.read_csv(file_path)
            avg_intra_sim = df["intra_sim"].mean()
            avg_cluster_sim = df["cluster_sim"].mean()
            diff = avg_intra_sim - avg_cluster_sim

            file_name = os.path.basename(file_path).split(".")[0]
            results.append(
                {
                    "file": file_name,
                    "avg_intra_sim": avg_intra_sim,
                    "avg_cluster_sim": avg_cluster_sim,
                    "diff": diff,
                }
            )
        else:
            print(f"文件不存在：{file_path}")

    print(f"处理结果：{results}")
    if results:
        conclusion_df = pd.DataFrame(results)
        conclusion_path = f"results/{model_type}/conclude.csv"
        try:
            os.makedirs(os.path.dirname(conclusion_path), exist_ok=True)
            conclusion_df.to_csv(conclusion_path, index=False)
            print(f"结论已保存到 {conclusion_path}")
        except Exception as e:
            print(f"保存结论文件时出错：{e}")
    else:
        print(f"没有结果可以保存")


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
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        choices=[
            "gcn",
            "gat",
            "graphSAGE",
            "OrderedGNN",
            "SGFormer",
            "sign",
            "gcn_hole",
            "gat_hole",
            "graphSAGE_hole",
            "OrderedGNN_hole",
            "SGFormer_hole",
            "sign_hole",
        ],
        help="GNN type",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = set_device(args.gpu_id)

    EMB_PATH = Path("./data")
    GRAPH_PATH = Path(f"./tmp/grasp/ignn/graphs")
    GRASP_PATH = EMB_PATH / f"{args.type}"
    GRASP_CSV = Path(f"./results/{args.type}.csv")

    grasp_df = pd.read_csv(GRASP_CSV)
    grasp_df["sampled_page_ids"] = grasp_df["sampled_page_ids"].apply(
        lambda x: x.strip("[]").replace("'", "").split()
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

        grasp_row = grasp_df[grasp_df["url"] == emb_file_name].iloc[0]
        sampled_page_ids = grasp_row["sampled_page_ids"]

        labels = torch.tensor(labels).to(device)
        intra_sim_grasp_a11y = intra_similarity(feature_dict, labels, "a11y_emb", device)
        intra_sim_grasp_text = intra_similarity(feature_dict, labels, "text_emb", device)
        intra_sim_grasp_image = intra_similarity(feature_dict, labels, "image_emb", device)

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
                "model": args.type,
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_a11y,
                "cluster_sim": cluster_sim_grasp_a11y,
            },
            csv_name=f"{args.type}/a11y_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": args.type,
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_text,
                "cluster_sim": cluster_sim_grasp_text,
            },
            csv_name=f"{args.type}/text_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": args.type,
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_image,
                "cluster_sim": cluster_sim_grasp_image,
            },
            csv_name=f"{args.type}/image_emb.csv",
        )
    print("开始生成结论文件...")
    calculate_averages_and_save_conclusion(args.type)

print("Done")
