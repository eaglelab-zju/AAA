"""grasp"""

# pylint:disable=line-too-long,wrong-import-position,invalid-name
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from the_utils import make_parent_dirs, save_to_csv_files, set_device, set_seed

from grasp.distance import get_distance
from grasp.gen_graph import gen_graphs, get_rule_ids, load_bert_model, load_vit_model
from grasp.train_hole import train_hole

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
    args = parser.parse_args()

    set_seed(args.seed)
    device = set_device(str(args.gpu_id))
    mp.set_start_method("spawn", force=True)

    # DOM & screenshot: ./data/feature_test/UIST_DOMData/{folders}/{hctask_id}-xxx/{page_id}(.jpg)
    # axe results: ./data/feature_test/UIST_axeData/axe_res/{hctask_id}/{page_id}
    # adj: ./data/feature_test/UIST_GraphData/{hctask_id}/adj_matrix.txt
    base_path = Path("../data/TPS")
    GRAPH_DIR = Path("./tmp/ignn/graphs/")
    EMB_DIR = Path("data/grasp/ignn")
    dom_path = base_path.joinpath("UIST_DOMData")
    folders = [f for f in os.listdir(dom_path) if dom_path.joinpath(f).is_dir()]
    N_CLUSTERS = 20

    rule_ids = get_rule_ids()
    tokenizer, bert = load_bert_model()
    bert = bert.to(device)
    bert.eval()
    vit = load_vit_model().to(device)
    vit.eval()

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
            graph_path,
            subfolder,
            subfolder_name,
            base_path,
            rule_ids,
            tokenizer,
            bert,
            vit,
            device,
        )
        graph = graph.to(torch.device("cpu"))
        features = graph.ndata["feat"]
        elapsed_time_gen = time.time()

        if emb_path.exists():
            z_detached, q, c = torch.load(emb_path, map_location=device)
        else:
            make_parent_dirs(emb_path)
            z_detached, q, c = train_hole(folder, graph, features, N_CLUSTERS, device, "ignn")
            torch.save((z_detached, q, c), emb_path, pickle_protocol=4)

        z_to_c = torch.cdist(z_detached, c, p=2)
        nearest_nodes = torch.argmin(z_to_c, dim=0)
        sampled_page_ids = np.array(page_ids)[nearest_nodes.cpu().numpy().astype(int)]
        elapsed_time_hole = time.time()
        print("Sampled page IDs:", sampled_page_ids)

        # dis = get_distance(sampled_page_ids, subfolder)
        # save_to_csv_files(
        #     results={"TED": dis["TED"], "LEV": dis["LEV"], "TEDs": dis["TEDs"], "LEVs": dis["LEVs"]},
        #     csv_name="grasp_dis.csv",
        #     append_info={"sampled pages": sampled_page_ids, "time": f"{elapsed_time:.2f}s"},
        # )
        print(f"{folder} Overall Time:{elapsed_time_hole-start_time:6.2f}s\n\n")

        save_to_csv_files(
            results={
                "site": folder,
                "url": subfolder_name,
                "clustering": q,
                "sampled_page_ids": sampled_page_ids,
                "gen_elapsed_time": f"{elapsed_time_gen-start_time:6.2f}",
                "hole_elapsed_time": f"{elapsed_time_hole-start_time:6.2f}",
            },
            csv_name="ignn/grasp.csv",
        )
