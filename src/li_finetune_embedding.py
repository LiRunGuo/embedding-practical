import os
from dotenv import load_dotenv
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


TRAIN_CORPUS_FPATH = os.getenv("TRAIN_CORPUS_FPATH", "./train_corpus.json")
VAL_CORPUS_FPATH = os.getenv("VAL_CORPUS_FPATH", "./val_corpus.json")


def main():
    load_dotenv()
    train_dataset = EmbeddingQAFinetuneDataset.from_json(TRAIN_CORPUS_FPATH)
    val_dataset = EmbeddingQAFinetuneDataset.from_json(VAL_CORPUS_FPATH)
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id="BAAI/bge-small-zh-v1.5",
        model_output_path="zhengquan",
        val_dataset=val_dataset,
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    print(embed_model)


if __name__ == "__main__":
    main()


