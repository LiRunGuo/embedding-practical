import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.finetuning import generate_qa_embedding_pairs


# 使用前，请将 PDF 路径改为你本机的路径
BASE_DIR = os.getenv("EMBED_PDF_BASE", "./data/")
TRAIN_FILES = [os.path.join(BASE_DIR, "中华人民共和国证券法(2019修订).pdf")]
TRAIN_CORPUS_FPATH = os.getenv("TRAIN_CORPUS_FPATH", "./train_corpus.json")
VAL_CORPUS_FPATH = os.getenv("VAL_CORPUS_FPATH", "./val_corpus.json")


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    return nodes


def main():
    load_dotenv()
    nodes = load_corpus(TRAIN_FILES, verbose=True)
    ollm = Ollama(model="qwen2:7b-instruct-q4_0", request_timeout=120.0)
    train_dataset = generate_qa_embedding_pairs(llm=ollm, nodes=nodes)
    val_dataset = generate_qa_embedding_pairs(llm=ollm, nodes=nodes)
    train_dataset.save_json(TRAIN_CORPUS_FPATH)
    val_dataset.save_json(VAL_CORPUS_FPATH)
    print(f"Saved train to {TRAIN_CORPUS_FPATH} and val to {VAL_CORPUS_FPATH}")


if __name__ == "__main__":
    main()


