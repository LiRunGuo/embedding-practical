## Embedding 技术实战

本项目依据提供 PyTorch 基础 Embedding、Word2Vec（Skip-gram）示例，以及基于 LlamaIndex 的 Embedding 微调、评估（Hit-Rate / MRR）与 MTEB 评测的可运行脚本。

---

### 目录结构

```
Embedding技术实战项目/
  ├─ README.md
  ├─ requirements.txt
  ├─ env.example
  └─ src/
      ├─ torch_embedding_demo.py
      ├─ word2vec_skipgram.py
      ├─ li_generate_qa_pairs.py
      ├─ li_finetune_embedding.py
      ├─ li_eval_hit_rate.py
      ├─ li_eval_sentence_transformers.py
      ├─ prepare_eu_ai_act_dataset.py
      ├─ mrr_eval_openai_and_hf.py
      └─ mteb_eval.py
```

---

### 环境准备

- Python 3.10（推荐）
- 建议：GPU + CUDA（SentenceTransformer 微调与大模型推理更高效）

安装依赖：
```
pip install -r requirements.txt
```

配置环境变量：
```
复制 env.example 为 .env 并按需填写
```
支持的变量：
- `HUGGINGFACE_HUB_TOKEN`：从 Hugging Face 拉取模型/数据
- `OPENAI_API_KEY`：如需使用 OpenAI Embedding
- `OLLAMA_BASE_URL`：如使用本地 Ollama 以生成 QA 对

---

### 快速开始

- PyTorch Embedding 演示：
```
python src/torch_embedding_demo.py
```

- Word2Vec（Skip-gram）示例训练：
```
python src/word2vec_skipgram.py
```

- 基于 LlamaIndex 从本地 PDF 生成 QA 对（需准备 PDF 路径，见脚本顶部注释）：
```
python src/li_generate_qa_pairs.py
```

- 使用 SentenceTransformers 微调（基于生成的 QA 数据集）：
```
python src/li_finetune_embedding.py
```

- 评估（Hit-Rate）：
```
python src/li_eval_hit_rate.py
```

- 评估（Sentence-Transformers 官方 InformationRetrievalEvaluator）：
```
python src/li_eval_sentence_transformers.py
```

- 生成欧盟 AI 法案 EN 语料 QA 数据集（Reverse HyDE 思路）：
```
python src/prepare_eu_ai_act_dataset.py
```

- 多 Embedding 模型 MRR 评测（OpenAI/HF开源）：
```
python src/mrr_eval_openai_and_hf.py
```

- 使用 MTEB 评测本地微调模型：
```
python src/mteb_eval.py
```

---

### 要点说明

- QA 对生成：示例使用 `generate_qa_embedding_pairs`（LlamaIndex），可用本地 Ollama（默认 `qwen2:7b-instruct-q4_0`）或 OpenAI 模型。Ollama 需提前安装并拉取模型，或切换到云端模型（见脚本注释）。
- 微调：采用 `SentenceTransformersFinetuneEngine`，底模默认 `BAAI/bge-small-zh-v1.5`。
- 评估：提供 Hit-Rate 与 MRR 两种评测方式；MTEB 可对微调后模型进行标准化测评。
- Windows 注意：若需 bitsandbytes 量化加速，需参考其 Windows 轮子兼容性说明（本项目默认不强制依赖）。

---

### 参考

- LlamaIndex 微调文档：`https://docs.llamaindex.ai/en/stable/use_cases/fine_tuning/`
- MTEB：`https://github.com/embeddings-benchmark/mteb`
- BGE 模型：`https://huggingface.co/BAAI/bge-small-zh-v1.5`


