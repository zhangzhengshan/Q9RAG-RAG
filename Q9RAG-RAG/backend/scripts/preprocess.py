import json
import os
import re
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GLOBAL_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ret_key(
    json_list: List[Dict[str, Any]],
    keep_keys=["url", "title", "publishedAt", "content"],
):
    filtered_list = []
    for item in json_list:
        filtered_item = {key: item[key] for key in keep_keys if key in item and item[key]}
        if "content" in filtered_item:
            filtered_list.append(filtered_item)
    return filtered_list


def manual_semantic_splitter(content: str, threshold: float = 0.45):
    sentences = re.split(r"(?<=[。！？!?])\s*|(?<=\n)", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return sentences

    embeddings = GLOBAL_MODEL.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    print(f"\n--- semantic split log (threshold: {threshold}) ---")

    for i in range(len(sentences) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        log_status = "keep" if sim >= threshold else "split"
        print(f"sentence {i} & {i + 1} | similarity: {sim:.4f} | status: {log_status}")

        if sim >= threshold:
            current_chunk.append(sentences[i + 1])
        else:
            chunks.append("".join(current_chunk))
            current_chunk = [sentences[i + 1]]

    chunks.append("".join(current_chunk))
    print(f"chunk count: {len(chunks)}\n")
    return chunks


def text_splitters(filtered_list: List[Dict[str, Any]], threshold: float = 0.45):
    result_chunks = []
    chunk_id = 0

    for article in filtered_list:
        print(f"processing: {article.get('title')}")
        content_chunks = manual_semantic_splitter(article["content"], threshold=threshold)

        for idx, chunk_content in enumerate(content_chunks):
            result_chunks.append(
                {
                    "id": chunk_id,
                    "url": article["url"],
                    "title": article["title"],
                    "publishedAt": article["publishedAt"],
                    "content_chunk": chunk_content,
                    "chunk_idx": idx + 1,
                    "total_chunks": len(content_chunks),
                }
            )
            chunk_id += 1
    return result_chunks


def embed_with_faiss(result_chunks: List[Dict[str, Any]]):
    if not result_chunks:
        return None, []

    content_chunks = [c["content_chunk"] for c in result_chunks]
    ids = np.array([c["id"] for c in result_chunks], dtype=np.int64)

    embeddings = GLOBAL_MODEL.encode(
        content_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    base_index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    base_index.hnsw.efConstruction = 200

    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, "vector_index.bin")
    with open("metadatatwo.json", "w", encoding="utf-8") as f:
        json.dump(result_chunks, f, ensure_ascii=False, indent=2)

    print("FAISS index and metadata saved.")
    return index, result_chunks


if __name__ == "__main__":
    input_json_path = r"E:\图片\豆包\Q9RAG-RAG\Q9RAG-RAG\data\ps_2026-06-19_all.json"
    raw_data = read_json(input_json_path)
    filtered_data = ret_key(raw_data)

    chunks = text_splitters(filtered_data, threshold=0.45)
    embed_with_faiss(chunks)
