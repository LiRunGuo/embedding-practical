from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from pathlib import Path


def evaluate_st(dataset: EmbeddingQAFinetuneDataset, model_id: str, name: str):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    evaluator_st = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator_st(model, output_path=output_path)


def main():
    val_dataset = EmbeddingQAFinetuneDataset.from_json("./val_corpus.json")
    evaluate_st(val_dataset, "BAAI/bge-small-zh-v1.5", name="bge-zh")
    evaluate_st(val_dataset, "zhengquan", name="finetuned")


if __name__ == "__main__":
    main()


