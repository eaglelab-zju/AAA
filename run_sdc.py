"""SDC"""

import argparse
import os
import time
from pathlib import Path

from the_utils import save_to_csv_files, set_device, set_seed

from sdc import SDCAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SDC",
        description="SDC",
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

    base_path = Path("./data/feature_test")
    dom_path = base_path.joinpath("UIST_DOMData")
    folders = [f for f in os.listdir(dom_path) if dom_path.joinpath(f).is_dir()]
    N_CLUSTERS = 20

    for folder in folders:
        start_time = time.time()

        folder_path = dom_path.joinpath(folder)
        subfolder_name = next(
            (d for d in os.listdir(folder_path) if folder_path.joinpath(d).is_dir()),
            None,
        )
        subfolder = folder_path.joinpath(subfolder_name)

        results = SDCAnalyzer(device=device).cluster_sampling(subfolder)

        elapsed_time = time.time() - start_time
        print(f"{folder} Overall Time:{elapsed_time:6.2f}s\n\n")

        save_to_csv_files(
            results={"site": folder, "url": subfolder_name, **results},
            append_info={"overall_elapsed_time": elapsed_time},
            csv_name="sdc.csv",
        )
