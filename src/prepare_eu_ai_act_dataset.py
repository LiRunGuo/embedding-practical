from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms.ollama import Ollama


def prepare_data():
    language = "EN"
    url_doc = (
        "https://eur-lex.europa.eu/legal-content/" + language + "/TXT/HTML/?uri=CELEX:32024R1689"
    )
    documents = SimpleWebPageReader(html_to_text=True).load_data([url_doc])
    parser = SentenceSplitter(chunk_size=1000)
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    return nodes


def main():
    nodes = prepare_data()
    prompts = {}
    prompts["EN"] = (
        "Context information is below.\n" "---------------------\n" "{context_str}\n" "---------------------\n"
        "Given the context information and not prior knowledge, generate only questions based on the below query.\n"
        "You are a Teacher/ Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination.\n"
        "The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
    )
    ollm = Ollama(model="qwen2:7b-instruct-q4_0", request_timeout=120.0)
    qa_dataset = generate_qa_embedding_pairs(
        llm=ollm,
        nodes=nodes,
        qa_generate_prompt_tmpl=prompts["EN"],
        num_questions_per_chunk=2,
    )
    qa_dataset.save_json("EN_dataset.json")
    print("Saved EN_dataset.json")


if __name__ == "__main__":
    main()


