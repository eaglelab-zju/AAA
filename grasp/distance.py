"""Distance"""

# pylint: disable=broad-exception-caught,too-many-locals
import itertools
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from random import shuffle

import apted
import numpy as np
from apted.helpers import Tree
from bs4 import BeautifulSoup
from rapidfuzz import distance


def extract_dom_text(html_path):
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error extracting text from {html_path}: {e}")
        return ""


def build_tree(element):
    """build tree"""
    if not element or element.name is None:
        return Tree("empty")
    node = Tree(element.name)
    for child in element.children:
        if child.name is not None:
            node.children.append(build_tree(child))
    return node


def html_to_tree(html_path):
    """convert html into tree"""
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    root = soup.body or soup.html
    tree = build_tree(root) if root else Tree("empty")
    return tree


def process_html(page_id, subfolder):
    """preprocess html"""
    html_path = os.path.join(subfolder, page_id)
    try:
        tree = html_to_tree(html_path)
        text = extract_dom_text(html_path)
        return page_id, tree, text
    except Exception:
        traceback.print_exc()
        print(f"failed to  process_html [{page_id}]", flush=True)
        return page_id, None, None


def compute_distances(page1, page2, trees, texts):
    """compute distances"""
    ted, lev = None, None
    t = time.time()
    if page1 in trees and page2 in trees and trees[page1] and trees[page2]:
        try:
            ted = apted.APTED(trees[page1], trees[page2]).compute_edit_distance()
        except Exception:
            traceback.print_exc()
            print(f"TED failed [{page1} vs {page2}]", flush=True)

    if page1 in texts and page2 in texts and texts[page1] and texts[page2]:
        try:
            lev = distance.Levenshtein.distance(texts[page1], texts[page2])
        except Exception:
            print(f"LEV failed [{page1} vs {page2}]")

    print(f"{page1:10} {page2:10} \tTED:{ted:8} LEV:{lev:8} Time:{time.time()-t:6.2f}s", flush=True)
    return page1, page2, ted, lev


def get_distance(sampled_page_ids, subfolder):
    """representativeness metrics
    1. HTML structure of Tree Edit Distance (TED) for layout representativeness
    2. DOM content of Levenshtein Distance (LEV) for content representativeness
    """
    t = time.time()

    shuffle(sampled_page_ids)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_html, sampled_page_ids, [subfolder] * len(sampled_page_ids))

    trees = {}
    texts = {}
    for page_id, tree, text in results:
        trees[page_id] = tree
        texts[page_id] = text

    all_pairs = list(itertools.combinations(sampled_page_ids, 2))
    shuffle(all_pairs)
    print(f"Process html done. {len(all_pairs)} pairs...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(
            compute_distances,
            [p1 for p1, p2 in all_pairs],
            [p2 for p1, p2 in all_pairs],
            [trees for _ in all_pairs],
            [texts] * len(all_pairs),
            chunksize=4,
        )

    ted_distances = []
    lev_distances = []
    for _, _, ted, lev in results:
        ted_distances.append(ted)
        lev_distances.append(lev)

    ted_avg = np.mean(ted_distances)
    ted_std = np.std(ted_distances)
    lev_avg = np.mean(lev_distances)
    lev_std = np.std(lev_distances)
    print(f"TED:{ted_avg:8.2f} LEV:{lev_avg:8.2f} Time:{time.time() - t:6.2f}s")

    return {
        "TED": f"{ted_avg:.2f}±{ted_std:.2f}",
        "LEV": f"{lev_avg:.2f}±{lev_std:.2f}",
        "TEDs": ted_distances,
        "LEVs": lev_distances,
    }
