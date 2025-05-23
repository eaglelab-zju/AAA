import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from the_utils import make_parent_dirs, save_to_csv_files, set_device, set_seed

sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../gnn")
sys.path.append(f"{os.path.abspath(os.path.dirname(__file__))}/../grasp")
from gen_graph import gen_graphs, get_rule_ids, load_bert_model, load_vit_model
from train_gnn import train_gnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GRASP",
        description="GRASP",
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
        choices=["gcn", "gat", "graphSAGE", "sign", "OrderedGNN", "SGFormer"],
        help="GNN type",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = set_device(str(args.gpu_id))
    mp.set_start_method("spawn", force=True)

    base_path = Path("../../data")
    GRAPH_DIR = Path(f"./tmp/{args.type}/graphs/")
    EMB_DIR = Path(f"data/{args.type}")
    dom_path = base_path.joinpath("UIST_DOMData")
    folders = [f for f in os.listdir(dom_path) if dom_path.joinpath(f).is_dir()]
    N_CLUSTERS = 20

    rule_ids = get_rule_ids()
    tokenizer, bert = load_bert_model()
    bert = bert.to(device).eval()
    vit = load_vit_model().to(device).eval()

    for folder in folders:
        folder_path = dom_path.joinpath(folder)
        subfolder_name = next(
            (d for d in os.listdir(folder_path) if folder_path.joinpath(d).is_dir()),
            None,
        )
        subfolder = folder_path.joinpath(subfolder_name)
        graph_path = GRAPH_DIR.joinpath(folder, subfolder_name)
        emb_path = EMB_DIR.joinpath(folder, subfolder_name)

        start_time = time.time()
        graph, page_ids = gen_graphs(
            graph_path, subfolder, subfolder_name, base_path, rule_ids, tokenizer, bert, vit, device
        )
        graph = graph.cpu()
        features = graph.ndata["feat"]
        elapsed_time_gen = time.time()

        if emb_path.exists():
            z_detached, q, c = torch.load(emb_path, map_location=device)
        else:
            make_parent_dirs(emb_path)
            z_detached, q, c = train_gnn(folder, graph, features, N_CLUSTERS, device, args.type)
            torch.save((z_detached, q, c), emb_path, pickle_protocol=4)

        z_to_c = torch.cdist(z_detached, c, p=2)
        nearest_nodes = torch.argmin(z_to_c, dim=0)
        sampled_page_ids = np.array(page_ids)[nearest_nodes.cpu().numpy().astype(int)]
        elapsed_time_hole = time.time()
        print("Sampled page IDs:", sampled_page_ids)

        print(f"{folder} Overall Time:{elapsed_time_hole-start_time:6.2f}s\n\n")

        save_to_csv_files(
            results={
                "site": folder,
                "url": subfolder_name,
                "clustering": q,
                "sampled_page_ids": sampled_page_ids,
                "gen_time": f"{elapsed_time_gen-start_time:.2f}s",
                "total_time": f"{time.time()-start_time:.2f}s",
            },
            csv_name=f"{args.type}.csv",
        )
