from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "BAAI/bge-reranker-base",
    use_fp16=False
)

score = reranker.compute_score(
    ["你好", "你好"]
)

print(score)