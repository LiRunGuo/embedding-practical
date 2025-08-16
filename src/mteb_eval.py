from mteb import MTEB
from sentence_transformers import SentenceTransformer


def main():
    model_name = "zhengquan"  # 微调产物目录名
    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=["Banking77Classification"])  # 示例任务
    results = evaluation.run(model, output_folder=f"results/{model_name}")
    print(results)


if __name__ == "__main__":
    main()


