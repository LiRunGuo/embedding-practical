import os
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def evaluate(dataset, embed_model, top_k=5):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        eval_results.append(mrr)
    return float(np.average(eval_results))


def main():
    load_dotenv()
    qa_dataset = EmbeddingQAFinetuneDataset.from_json("EN_dataset.json")

    # OpenAI embedding models
    embeddings_model_spec = {
        "OAI-Large-256": {"model_name": "text-embedding-3-large", "dimensions": 256},
        "OAI-Large-3072": {"model_name": "text-embedding-3-large", "dimensions": 3072},
        "OAI-Small": {"model_name": "text-embedding-3-small", "dimensions": 1536},
        # 开源模型示例
        "BGE-Small-ZH": {"hf_model": "BAAI/bge-small-zh-v1.5"},
    }

    results = []
    for name, spec in embeddings_model_spec.items():
        if "model_name" in spec:
            embed_model = OpenAIEmbedding(model=spec["model_name"], dimensions=spec["dimensions"])  # type: ignore
        else:
            embed_model = HuggingFaceEmbedding(model_name=spec["hf_model"])  # type: ignore
        score = evaluate(qa_dataset, embed_model)
        results.append((name, score))

    for name, score in results:
        print(f"{name}: MRR={score:.4f}")


if __name__ == "__main__":
    main()


