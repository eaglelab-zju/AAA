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
    GRAPH_PATH = Path("./tmp/ignn/graphs")
    GRASP_PATH = EMB_PATH / "grasp" / "ignn"
    GRASP_CSV = Path("./results/ignn/grasp.csv")
    DOM_PATH = Path("../data/TPS/UIST_DOMData")

    print(GRASP_PATH)
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
                "model": "grasp_ignn",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_a11y,
                "cluster_sim": cluster_sim_grasp_a11y,
            },
            csv_name="ignn/a11y_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": "grasp_ignn",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_text,
                "cluster_sim": cluster_sim_grasp_text,
            },
            csv_name="ignn/text_emb.csv",
        )
        save_to_csv_files(
            results={
                "model": "grasp_ignn",
                "url": emb_file_name,
                "intra_sim": intra_sim_grasp_image,
                "cluster_sim": cluster_sim_grasp_image,
            },
            csv_name="ignn/image_emb.csv",
        )
    # Compute and save text/image averages and differences (4 decimal places)
    def _save_summary(input_csv: Path, output_csv: Path, model: str = "grasp"):
        try:
            if not input_csv.exists():
                print(f"Input file not found: {input_csv}")
                return
            df = pd.read_csv(input_csv)
            if df.empty:
                print(f"Input file is empty: {input_csv}")
                return
            avg_intra = df["intra_sim"].mean()
            avg_cluster = df["cluster_sim"].mean()
            diff = avg_intra - avg_cluster

            summary_df = pd.DataFrame(
                [
                    {
                        "model": model,
                        "avg_intra_sim": f"{avg_intra:.4f}",
                        "avg_cluster_sim": f"{avg_cluster:.4f}",
                        "diff": f"{diff:.4f}",
                    }
                ]
            )
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_csv, index=False)
            print(f"Saved summary: {output_csv}")
        except Exception as e:
            print(f"Error saving summary for {input_csv}: {e}")

    print("Generating text/image summaries...")
    _save_summary(Path("./results/ignn/text_emb.csv"), Path("./results/ignn/text_summary.csv"))
    _save_summary(Path("./results/ignn/image_emb.csv"), Path("./results/ignn/image_summary.csv"))

print("Done!")
