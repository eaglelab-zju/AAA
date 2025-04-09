"""SDC"""

# pylint: disable=broad-exception-caught,too-many-locals,invalid-name,unused-variable
import os
import pickle
import re
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict

import jieba
import numpy as np
import torch
from bs4 import BeautifulSoup
from fast_pytorch_kmeans import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from the_utils import make_parent_dirs

jieba.setLogLevel(20)


def json_dump(obj, path: Path):
    make_parent_dirs(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def json_load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_tree(html):
    tag_pattern = re.compile(r"</?(\w+)([^>]*)/?>")
    tag_tree = []
    stack = []
    for match in tag_pattern.finditer(html):
        tag_name = match.group(1)
        full_tag = match.group(0)
        is_self_closing = full_tag.endswith("/>")
        is_closing = full_tag.startswith("</")
        if is_closing:
            if stack and stack[-1] == tag_name:
                stack.pop()
            tag_tree.append(tag_name)
        elif is_self_closing:
            tag_tree.append(tag_name)
        else:
            stack.append(tag_name)
            tag_tree.append(tag_name)
    return tag_tree


def process_html_file(html_file_path):
    PATTERN = r"{|\"|'|:|,|\[|\]|}"
    try:
        t = time.time()
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        tags = [tag.name for tag in soup.find_all()]
        structure = []
        tree = get_tree(html_content)
        content = soup.get_text(separator=" ", strip=True).strip()
        # print(content)
        struc_cont = []

        for tag in soup.find_all():
            tag_attr = "".join(re.split(PATTERN, f"{tag.name} {str(tag.attrs)}")).strip()
            tag_content = "".join(list(jieba.cut(tag.text))).strip()
            tag_close = tag.name if not tag.is_empty_element else ""
            structure.extend(tag_attr.split(" "))
            struc_cont.extend(f"{tag_attr} {tag_content} {tag_close}".split(" "))

        print(f"{html_file_path.name} done in {time.time()-t:.2f}s", flush=True)
        return html_file_path.name, {
            "tags": tags,
            "structure": structure,
            "tree": tree,
            "content": [content],
            "struc_cont": struc_cont,
        }

    except Exception:
        print(f"Error processing {html_file_path}")
        traceback.print_exc()
        return html_file_path.name, None


class SDCAnalyzer:
    """Web Structure Derived Clustering for Optimized Web Accessibility Evaluation

    1. word token set pre-establishment ?
    2. standardization of representation construction
    """

    def __init__(self, device=torch.device("cpu")):
        self.vectorizer = CountVectorizer()
        self.tsne = TSNE(n_components=3, n_jobs=cpu_count())
        self.device = device

        self.page_ids = []
        self.all_data = {}
        self.repr = {}

    def merge_reprs(self, repr_path: Path = None):
        sorted_page_ids = sorted(self.all_data.keys())
        self.repr = {
            key: [" ".join(self.all_data[page][key]) for page in sorted_page_ids]
            for key in ["tags", "structure", "tree", "content", "struc_cont"]
        }
        self.repr["page_ids"] = sorted_page_ids
        json_dump(self.repr, repr_path)
        self.page_ids = self.repr.pop("page_ids")

    def extract_representations(
        self, site_dir: Path, save_dir: Path = Path("./data/sdc"), load=True
    ):
        repr_path = save_dir.joinpath("repr")
        t = time.time()

        if load and repr_path.exists():
            self.repr: Dict = json_load(repr_path)
            self.page_ids = self.repr.pop("page_ids")
            return f"{time.time()-t:.2f}"

        all_path = save_dir.joinpath("all")
        if load and all_path.exists():
            self.all_data = json_load(all_path)
            self.merge_reprs(repr_path)
            return f"{time.time()-t:.2f}"

        page_ids = [
            p for p in os.listdir(site_dir) if not p.lower().endswith((".jpg", ".logs", ".log"))
        ]

        SEP = "=" * 6
        print(f"\n\n{SEP}{site_dir.name} {len(page_ids)} start{SEP}")
        with Pool(cpu_count()) as pool:
            results = pool.map(process_html_file, [site_dir.joinpath(p) for p in page_ids])
        self.all_data = {page_id: data for page_id, data in results if data}
        json_dump(self.all_data, all_path)
        self.merge_reprs(repr_path)
        elapsed_time = f"{time.time()-t:.2f}"
        print(f"{SEP}{site_dir.name}end{SEP} in {elapsed_time}s\n\n")
        return elapsed_time

    def cluster_sampling(self, site_dir: Path, save_dir: Path = Path("./data/sdc"), n_clusters=20):
        save_dir = save_dir.joinpath(os.path.basename(site_dir))
        elapsed_time_e = self.extract_representations(site_dir, save_dir=save_dir)

        print(f"\n\n{os.path.basename(site_dir)} clustering begins....")
        results = {}
        for key, _ in self.repr.items():
            save_path = save_dir.joinpath(key)
            elapsed_time_k = 0
            elapsed_time_tsne_t = 0
            elapsed_time_tsne = 0

            if save_path.exists():
                (
                    dense_matrix,
                    tsne_matrix,
                    kmeans_clusters,
                    kmeans_clusters_c,
                    tsne_kmeans_clusters,
                    tsne_kmeans_clusters_c,
                ) = torch.load(save_path, map_location=self.device)

                z_to_c = torch.cdist(dense_matrix, kmeans_clusters_c, p=2)
                nearest_nodes = torch.argmin(z_to_c, dim=0)
                sampled_page_ids_k = np.array(self.page_ids)[
                    nearest_nodes.cpu().numpy().astype(int)
                ]

                z_to_c = torch.cdist(tsne_matrix, tsne_kmeans_clusters_c, p=2)
                nearest_nodes = torch.argmin(z_to_c, dim=0)
                sampled_page_ids_t = np.array(self.page_ids)[
                    nearest_nodes.cpu().numpy().astype(int)
                ]
            else:
                t = time.time()
                token_matrix = self.vectorizer.fit_transform(self.repr[key])
                dense_matrix = token_matrix.toarray()
                elapsed_time_token = time.time()

                tsne_matrix = torch.from_numpy(self.tsne.fit_transform(dense_matrix)).to(
                    self.device
                )
                dense_matrix = (
                    torch.from_numpy(token_matrix.toarray()).to(torch.float).to(self.device)
                )
                elapsed_time_tsne = f"{time.time()-elapsed_time_token:.2f}"
                print(f"\n{key} matrix dim: {dense_matrix.shape}")
                print(f"vectorizer {elapsed_time_token-t}s, tsne_matrix {elapsed_time_tsne}s")

                t = time.time()
                kmeans = KMeans(n_clusters=n_clusters, init_method="kmeans++")
                kmeans_clusters = kmeans.fit_predict(dense_matrix)
                kmeans_clusters_c = kmeans.centroids
                z_to_c = torch.cdist(dense_matrix, kmeans_clusters_c, p=2)
                nearest_nodes = torch.argmin(z_to_c, dim=0)
                sampled_page_ids_k = np.array(self.page_ids)[
                    nearest_nodes.cpu().numpy().astype(int)
                ]
                elapsed_time_k = f"{time.time()-t:.2f}"
                print(f"kmeans_clusters done in {elapsed_time_k}s")
                print(f"kmeans_clusters Sampled page IDs: {sampled_page_ids_k}")

                t = time.time()
                tsne_kmeans = KMeans(n_clusters=n_clusters, init_method="kmeans++")
                tsne_kmeans_clusters = tsne_kmeans.fit_predict(tsne_matrix)
                tsne_kmeans_clusters_c = tsne_kmeans.centroids
                z_to_c = torch.cdist(tsne_matrix, tsne_kmeans_clusters_c, p=2)
                nearest_nodes = torch.argmin(z_to_c, dim=0)
                sampled_page_ids_t = np.array(self.page_ids)[
                    nearest_nodes.cpu().numpy().astype(int)
                ]
                elapsed_time_tsne_t = f"{time.time()-t:.2f}"
                print(f"tsne_kmeans_clusters done in {elapsed_time_tsne_t}s")
                print(f"tsne_kmeans_clusters Sampled page IDs: {sampled_page_ids_t}")

                torch.save(
                    (
                        dense_matrix,
                        tsne_matrix,
                        kmeans_clusters,
                        kmeans_clusters_c,
                        tsne_kmeans_clusters,
                        tsne_kmeans_clusters_c,
                    ),
                    save_path,
                    pickle_protocol=4,
                )

            results = {
                **results,
                f"{key}_kmeans_c_assignments": kmeans_clusters,
                f"{key}_kmeans_c_sampled_page_ids": sampled_page_ids_k,
                f"{key}_elapsed_time_k": elapsed_time_k,
                f"{key}_tsne_kmeans_c_assignments": tsne_kmeans_clusters,
                f"{key}_tsne_kmeans_c_sampled_page_ids": sampled_page_ids_t,
                f"{key}_elapsed_time_tsne_t": elapsed_time_tsne_t,
                f"{key}_tsne_time": elapsed_time_tsne,
            }
        return {"extraction_time": elapsed_time_e, **results}
