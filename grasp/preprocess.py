"""Preprocessing"""

import cv2
import torch
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def gen_screenshot_embedding(image_path, vit, device):
    image = cv2.imread(image_path)
    with torch.no_grad():
        return vit(transform(image)[None, ...].to(device))


def extract_visible_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    visible_text = soup.get_text(separator=" ", strip=True)
    return visible_text


def get_bert_embeddings(texts, tokenizer, bert, device):
    interval = 256
    text_embeddings = []
    for i in range(0, len(texts), interval):
        with torch.no_grad():
            encoded_inputs = tokenizer(
                texts[i : i + interval],
                add_special_tokens=True,
                padding="max_length",
                max_length=10,
                return_tensors="pt",
                truncation=True,
            ).to(device)

            # [N, H]
            text_embedding = bert(**encoded_inputs).last_hidden_state[:, 0, :]
        text_embeddings.append(text_embedding.cpu())

    # N * 768 --> (N / 4) * (768 * 4)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = text_embeddings.reshape(
        -1, text_embeddings.shape[0], text_embeddings.shape[1]
    )
    text_embeddings, _ = torch.max(text_embeddings, dim=1)
    return text_embeddings.to(device)


def cluster_embeddings(embeddings_matrix, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings_matrix.numpy())
    return clusters


def gen_text_embedding(html_file, tokenizer, model, device, default=None, page_id=None):
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    visible_text = extract_visible_text(html_content)
    if len(visible_text) == 0:
        print(f"page {page_id} has no visible_text")
        return default
    return get_bert_embeddings(visible_text, tokenizer, model, device)
