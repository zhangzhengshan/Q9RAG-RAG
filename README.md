**• 核心实现与技术亮点：**

**• 混合检索架构优化**：采用BM25+FAISSHybrid Retrieval 方案，同时利用关键词匹配能力与语义向量召回能力，缓解纯向量检索存在的关键词遗漏问题；使用BAAI/bge-small-zh-v1.5生成文本Embedding，通过Top-K多路召回策略提升知识覆盖率。

**• 多粒度检索与重排序**：针对长文本设计Parent-Child Retriever 双索引结构，在Chunking阶段平衡检索粒度；构建“BM25召回+向量召回+BGE-Reranker重排序”三级检索Pipeline，对候选文档进行精细化筛选，有效降低无关内容进入上下文窗口的概率。

**• 精细化RAG Pipeline设计**：独立设计Query→Hybrid Retrieval →Re-Rank→Context Assembly→LLM Generation全链路流程；引入Lost in the Middle优化策略与上下文窗口控制机制，提升长文档场景下的关键信息利用率。

**• 量化成果**：独立编写~20个核心脚本文件，实现~2000行干净规范的Python代码；引入父子索引与BGE-Reranker 后，检索准确率（Hit Rate@5）提升50%，回答准确率显著提升；通过拒答机制拦截了30%范围外的无关恶意提问。
