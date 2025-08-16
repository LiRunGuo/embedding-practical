from typing import List, Dict
from tqdm import tqdm
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def evaluate(dataset: EmbeddingQAFinetuneDataset, embed_model, top_k: int = 5):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_rows = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids
        eval_rows.append({
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        })
    return pd.DataFrame(eval_rows)


def main():
    dataset = EmbeddingQAFinetuneDataset.from_json("./val_corpus.json")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    df = evaluate(dataset, embed_model)
    print(df)
    print("Hit Rate:", df["is_hit"].mean())


if __name__ == "__main__":
    main()


