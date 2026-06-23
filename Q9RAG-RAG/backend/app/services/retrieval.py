import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
import os

import faiss
import numpy as np
import torch
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer


class Retriever:
    _instance = None


    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retrieval engine.

        The public retrieve() API stays unchanged, but the first-stage recall can
        run vector search, BM25 keyword search, or a hybrid of both.
        """
        self.config = config

        torch.set_num_threads(config.get("torch_threads", 6))

        self.model = SentenceTransformer(
            config["embedding_model_path"],
            device=config.get("device", "cpu"),
        )
        self.reranker = FlagReranker(
            config["reranker_model_path"],
            use_fp16=config.get("use_fp16", False),
        )

        self.metadata = self._load_metadata(config["metadata_path"])
        self.metadata_dict = {int(m["id"]): m for m in self.metadata}
        self.index = self._load_index(config["index_path"])
        self._set_faiss_search_params()

        self._build_bm25_index()

        print(
            f"[RAG] Retriever ready. Loaded {len(self.metadata)} chunks. "
            f"mode={self.config.get('retrieval_mode', 'hybrid')}"
        )

    @staticmethod
    def _load_metadata(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_index(path: str) -> faiss.Index:
        return faiss.read_index(path)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def _set_faiss_search_params(self) -> None:
        ef_search = self.config.get("faiss_ef_search")
        if ef_search is None:
            return

        base_index = self.index
        if isinstance(base_index, faiss.IndexIDMap):
            base_index = base_index.index

        if hasattr(base_index, "hnsw"):
            base_index.hnsw.efSearch = int(ef_search)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Dependency-free tokenizer for mixed Chinese/English content.

        Chinese text gets both character unigrams and adjacent bigrams. This
        keeps exact term recall usable without requiring jieba on the target PC.
        """
        if not text:
            return []

        tokens: List[str] = []
        parts = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        for part in parts:
            if re.fullmatch(r"[a-z0-9]+", part):
                tokens.append(part)
                continue

            chars = list(part)
            tokens.extend(chars)
            if len(chars) > 1:
                tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))

        return tokens

    def _build_bm25_index(self) -> None:
        self.bm25_k1 = float(self.config.get("bm25_k1", 1.5))
        self.bm25_b = float(self.config.get("bm25_b", 0.75))
        self.bm25_inverted: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.bm25_doc_lengths: Dict[int, int] = {}
        self.bm25_idf: Dict[str, float] = {}

        for meta in self.metadata:
            doc_id = int(meta["id"])
            title = meta.get("title") or ""
            content = meta.get("content_chunk") or ""
            tokens = self._tokenize(f"{title} {title} {content}")
            self.bm25_doc_lengths[doc_id] = len(tokens)

            for term, freq in Counter(tokens).items():
                self.bm25_inverted[term].append((doc_id, freq))

        doc_count = len(self.bm25_doc_lengths)
        total_length = sum(self.bm25_doc_lengths.values())
        self.bm25_avgdl = total_length / doc_count if doc_count else 0.0

        for term, postings in self.bm25_inverted.items():
            df = len(postings)
            self.bm25_idf[term] = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    @torch.inference_mode()
    def _vector_search(self, query: str) -> Dict[int, float]:
        top_k = int(self.config.get("top_k_initial", 5))
        threshold = float(self.config.get("sim_threshold", 0.45))
        current_count = int(self.index.ntotal)
        if top_k <= 0 or current_count <= 0:
            return {}

        query_embedding = np.asarray(
            self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        scores, ids = self.index.search(query_embedding, k=min(top_k, current_count))

        results: Dict[int, float] = {}
        for doc_id, score in zip(ids[0], scores[0]):
            doc_id = int(doc_id)
            score = float(score)
            if doc_id >= 0 and score >= threshold and doc_id in self.metadata_dict:
                results[doc_id] = score
        return results

    def _bm25_search(self, query: str) -> Dict[int, float]:
        top_k = int(self.config.get("top_k_bm25", 8))
        if top_k <= 0 or not self.bm25_avgdl:
            return {}

        query_terms = Counter(self._tokenize(query))
        if not query_terms:
            return {}

        scores: Dict[int, float] = defaultdict(float)
        for term, query_freq in query_terms.items():
            postings = self.bm25_inverted.get(term)
            if not postings:
                continue

            idf = self.bm25_idf.get(term, 0.0)
            for doc_id, term_freq in postings:
                doc_len = self.bm25_doc_lengths.get(doc_id, 0)
                denom = term_freq + self.bm25_k1 * (
                    1 - self.bm25_b + self.bm25_b * doc_len / self.bm25_avgdl
                )
                scores[doc_id] += (
                    query_freq
                    * idf
                    * (term_freq * (self.bm25_k1 + 1))
                    / max(denom, 1e-9)
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return dict(ranked[:top_k])

    def _merge_candidates(
        self,
        vector_scores: Dict[int, float],
        bm25_scores: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        vector_weight = float(self.config.get("vector_weight", 0.7))
        bm25_weight = float(self.config.get("bm25_weight", 0.3))
        rrf_k = float(self.config.get("rrf_k", 60))
        pool_size = int(
            self.config.get(
                "hybrid_pool_size",
                len(vector_scores) + len(bm25_scores),
            )
        )

        vector_rank = {
            doc_id: rank
            for rank, (doc_id, _) in enumerate(
                sorted(vector_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }
        bm25_rank = {
            doc_id: rank
            for rank, (doc_id, _) in enumerate(
                sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }

        candidates = []
        for doc_id in set(vector_scores) | set(bm25_scores):
            meta = self.metadata_dict.get(doc_id)
            if not meta:
                continue

            hybrid_score = 0.0
            if doc_id in vector_rank:
                hybrid_score += vector_weight / (rrf_k + vector_rank[doc_id])
            if doc_id in bm25_rank:
                hybrid_score += bm25_weight / (rrf_k + bm25_rank[doc_id])

            source = []
            if doc_id in vector_scores:
                source.append("vector")
            if doc_id in bm25_scores:
                source.append("bm25")

            candidate = meta.copy()
            candidate["_vector_score"] = vector_scores.get(doc_id)
            candidate["_bm25_score"] = bm25_scores.get(doc_id)
            candidate["_hybrid_score"] = hybrid_score
            candidate["_retrieval_source"] = "+".join(source)
            candidates.append(candidate)

        candidates.sort(key=lambda item: item["_hybrid_score"], reverse=True)
        return candidates[:pool_size] if pool_size > 0 else candidates

    def _first_stage_recall(self, query: str) -> List[Dict[str, Any]]:
        mode = str(self.config.get("retrieval_mode", "hybrid")).lower()
        if mode not in {"hybrid", "vector", "bm25"}:
            mode = "hybrid"

        vector_scores = self._vector_search(query) if mode in {"hybrid", "vector"} else {}
        bm25_scores = self._bm25_search(query) if mode in {"hybrid", "bm25"} else {}
        return self._merge_candidates(vector_scores, bm25_scores)

    @torch.inference_mode()
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        score_filter = float(self.config.get("score_filter", -0.7))
        final_n = int(self.config.get("final_n", 3))

        initial_candidates = self._first_stage_recall(query)
        if not initial_candidates:
            return []

        pairs = [[query, c["content_chunk"]] for c in initial_candidates]
        rerank_scores = self.reranker.compute_score(pairs)
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]

        valid_results = []
        for idx, score in enumerate(rerank_scores):
            if score <= score_filter:
                continue

            candidate = initial_candidates[idx]
            valid_results.append(
                {
                    "id": int(candidate.get("id")),
                    "title": candidate.get("title"),
                    "content_chunk": candidate.get("content_chunk"),
                    "rerank_score": float(score),
                    "retrieval_source": candidate.get("_retrieval_source"),
                    "vector_score": candidate.get("_vector_score"),
                    "bm25_score": candidate.get("_bm25_score"),
                    "hybrid_score": candidate.get("_hybrid_score"),
                }
            )

        return sorted(valid_results, key=lambda x: x["rerank_score"], reverse=True)[:final_n]
if __name__ == "__main__":
    config = {
        "metadata_path": r"E:\picture\doubao\Q9RAG-RAG\Q9RAG-RAG\backend\scripts\metadatatwo.json",
        "index_path": r"E:\picture\doubao\Q9RAG-RAG\Q9RAG-RAG\backend\scripts\vector_index.bin",
        "embedding_model_path": "BAAI/bge-small-zh-v1.5",
        "reranker_model_path": "BAAI/bge-reranker-base",
        "torch_threads": 4,
        "sim_threshold": 0.45,
        "top_k_initial": 5,
        "final_n": 3,
        "score_filter": -0.7
    }
    print("\n" + "=" * 50)
    print("正在初始化检索器...")
    retriever = Retriever.from_config(config)
    query_1 = "哲学是什么啊"
    print(f"\n[测试 1] 查询语句: '{query_1}'")
    results_1 = retriever.retrieve(query_1)

    for rank, res in enumerate(results_1, 1):
        print(f"排名 {rank}:")
        print(f"  ID: {res['id']} | 标题: {res['title']}")
        print(f"  召回源: {res['retrieval_source']}")
        print(f"  重排得分: {res['rerank_score']:.4f}")
        print(f"  混合召回得分: {res['hybrid_score']:.4f}")
        print(f"  片段内容: {res['content_chunk']}")

