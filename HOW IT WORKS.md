# Complete Explanation of the RAG Code Logic

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ INGESTION│ → │EMBEDDING │ → │ RETRIEVAL │ → │ GENERATION  │ │
│  │ Pipeline │   │ + Store  │   │  (Search) │   │   (LLM)     │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘ │
│                                                                 │
│  Phase 1: Input Data   Phase 2: Save Vectors   Phase 3: Answer│
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: Data Ingestion Pipeline

### Step 1 — Load Documents

```python
loader = TextLoader("spesifikasi_aleleon.txt")
docs = loader.load()
```

**What happens:**
- `TextLoader` reads raw `.txt` files into `Document` objects.
- Each `Document` has 2 attributes:
  - `page_content`: text content
  - `metadata`: file info (name, path)
- At this stage, **the entire file = 1 large document**.

```
spesifikasi_aleleon.txt (4076 chars)
        │
        ▼
┌──────────────────────────┐
│ Document(                │
│   page_content="Here     │
│   is the conversion..."  │
│   metadata={source:...}  │
│ )                        │
└──────────────────────────┘
```

### Step 2 — Chunking (Text Splitting)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n---", "\n## ", "\n\n", "\n", " "],
)
splits = text_splitter.split_documents(docs)
```

**What happens:**

LLMs have a context window limit. We cannot input the entire document at once. So the text is cut into small **chunks**.

**`RecursiveCharacterTextSplitter` Strategy:**

This splitter tries to cut **hierarchically** — starting from the largest separator down to the smallest:

```
Split priority:
  1. "\n---"   ← Split at horizontal rules (section dividers)
  2. "\n## "   ← Split at Markdown level 2 headings
  3. "\n\n"    ← Split at empty paragraphs
  4. "\n"      ← Split at new lines
  5. " "       ← Split at spaces (last resort)
```

**Why "recursive"?** Because if splitting at `\n---` results in a chunk > 1000 chars, the splitter **descends to the next level** (`\n## `), and so on until every chunk is ≤ 1000 chars.

**Parameters:**

```
chunk_size=1000    → Maximum 1000 characters per chunk
chunk_overlap=200  → The last 200 characters of chunk N are repeated at the start of chunk N+1
```

**Why overlap?** To ensure information at the chunk boundaries is not lost:

```
Original document:
"...RAM per node: 500GB effective | GPU per node: 2x RTX 3090..."

Without overlap (chunk_overlap=0):
  Chunk 1: "...RAM per node: 500GB effec"  ← cut off!
  Chunk 2: "tive | GPU per node: 2x RTX 3090..."

With overlap (chunk_overlap=200):
  Chunk 1: "...RAM per node: 500GB effective | GPU per no"
  Chunk 2: "500GB effective | GPU per node: 2x RTX 3090..."
                ↑ overlap — info is preserved
```

**Your document chunking results:**

```
Document (4076 chars)
        │
        ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Chunk 0 │ │ Chunk 1 │ │ Chunk 2 │ │ Chunk 3 │ │ Chunk 4 │ │ Chunk 5 │ │ Chunk 6 │
│ 303 chr │ │ 778 chr │ │ 958 chr │ │  3 chr  │ │ 976 chr │ │ 907 chr │ │ 151 chr │
│ Intro   │ │ Compute │ │ Interac │ │  "---"  │ │Software │ │ PkgMgr  │ │ Footer  │
│ ALELEON │ │  Node   │ │  Node   │ │         │ │ System  │ │ & Tools │ │         │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## PHASE 2: Embedding + Vector Database

### Step 3 — Vector Embedding

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Model used: `all-MiniLM-L6-v2`**

| Property | Detail |
|---|---|
| Architecture | MiniLM (distilled BERT) |
| Parameters | 22.7M |
| Output Dimensions | **384 dimensions** |
| Max Sequence | 256 tokens |
| Trained on | 1 Billion sentence pairs |
| Runs on | **CPU** (lightweight, ~90MB) |

**What is an embedding?**

An embedding converts text into a **vector of numbers** in a 384-dimensional space. Texts with **similar meanings** will have vectors that are **close** to each other.

```
"RAM per node in the epyc-jumbo partition is 500GB"
        │
        ▼  all-MiniLM-L6-v2
[0.032, -0.118, 0.245, ..., 0.067]    ← 384 numbers

"How much RAM per node is in the epyc-jumbo partition?"
        │
        ▼  all-MiniLM-L6-v2
[0.029, -0.121, 0.238, ..., 0.071]    ← 384 numbers (SIMILAR!)

"Does ALELEON support Docker?"
        │
        ▼  all-MiniLM-L6-v2
[-0.156, 0.089, -0.034, ..., 0.193]   ← 384 numbers (DISTANT!)
```

**This is NOT TF-IDF or BM25.**

| Method | How it works | Used in this code? |
|---|---|---|
| **TF-IDF** | Counts word frequency. "RAM" appearing 3x = relevant. Doesn't understand meaning. | ❌ |
| **BM25** | Advanced TF-IDF with document length normalization. | ❌ |
| **Sparse Retrieval** | Large vectors, mostly zeros. Matches keywords. | ❌ |
| **Dense Retrieval** ✅ | Text → dense 384D vector via neural network. Matches **meaning**. | ✅ **Used here** |

**Advantages of Dense Retrieval:**

```
Query: "I need a lot of memory for my job"
  │
  ├── TF-IDF/BM25: Search for word "memory" → NOT FOUND (document says "RAM")
  │
  └── Dense (MiniLM): Understands "memory" ≈ "RAM" semantically → FOUND ✅
```

### Step 4 — Vector Database (Chroma)

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
```

**What happens:**

1. Each chunk is embedded into a 384D vector.
2. The vector + original text is stored in the Chroma database (in-memory).

```
Chroma DB (in-memory)
┌────────────────────────────────────────────────────┐
│ ID │ Vector (384D)              │ Original Text    │
├────┼────────────────────────────┼──────────────────┤
│ 0  │ [0.03, -0.12, 0.24, ...]  │ "Here is the     │
│    │                            │  conversion..."  │
│ 1  │ [0.08, -0.05, 0.19, ...]  │ "## Compute Node │
│    │                            │  runs..."        │
│ 2  │ [-0.07, 0.14, 0.03, ...]  │ "## Interactive   │
│    │                            │  Node..."        │
│ ...│ ...                        │ ...              │
└────┴────────────────────────────┴──────────────────┘
```

**Chroma** is a vector database that is:
- Lightweight, runs **in-memory** (no separate server needed).
- Supports **cosine similarity search**.
- Suitable for prototyping (production usually uses Pinecone, Weaviate, Milvus).

---

## PHASE 3: Retrieval + Generation

### Retrieval — Search Relevan Chunks

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

**Retrieval type: Approximate Nearest Neighbor (ANN) with Cosine Similarity**

When a user asks a question, the process is:

```
User: "How much RAM in epyc-jumbo partition?"
         │
         ▼ all-MiniLM-L6-v2
Query Vector: [0.029, -0.121, 0.238, ..., 0.071]
         │
         ▼ Cosine Similarity against ALL chunks
         │
┌────────┬──────────────────────────────────┬────────────┐
│ Chunk  │ Content                          │ Similarity │
├────────┼──────────────────────────────────┼────────────┤
│ 1      │ "Compute Node... RAM 500GB..."   │ 0.87 ← #1 │
│ 2      │ "Interactive Node... RAM 60GB.." │ 0.72 ← #2 │
│ 4      │ "System Software..."             │ 0.41 ← #3 │
│ 0      │ "Intro ALELEON..."               │ 0.38 ← #4 │
│ 5      │ "Package Manager..."             │ 0.35 ← #5 │
│ 3      │ "---"                            │ 0.05       │
│ 6      │ "Footer..."                      │ 0.03       │
└────────┴──────────────────────────────────┴────────────┘
         │
         ▼ Get Top-K (k=5)
    Chunk 1, 2, 4, 0, 5 → sent to LLM as context
```

**Cosine Similarity Formula:**

```
                    A · B           Σ(Aᵢ × Bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ||A|| × ||B||     √Σ(Aᵢ²) × √Σ(Bᵢ²)

Result: -1 (opposite) to +1 (identical)
```

### Prompt Template — ChatML Format

```python
template_qwen = """<|im_start|>system
You are an AI agent, an assistance to the HPC Slurm admin...
Do not make up answers.<|im_end|>
<|im_start|>user
Reference Documents:
{context}

Question: {input}<|im_end|>
<|im_start|>assistant
"""
```

**Why the `<|im_start|>` / `<|im_end|>` format?**

This is the **ChatML format** — the format used during Qwen model training to distinguish roles:

```
<|im_start|>system     ← Instructions for the model (persona, rules)
...<|im_end|>
<|im_start|>user       ← User input
...<|im_end|>
<|im_start|>assistant   ← Model starts generating from here
```

**Template variables:**
- `{context}` → Automatically filled by LangChain with the 5 retrieved chunks.
- `{input}` → Filled with the user's question.

### Generation — vLLM + Qwen2.5

```python
llm = VLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
    vllm_kwargs={
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "max_model_len": 4096,
    }
)
```

**Generation flow:**

```
┌─────────────────────────────────────────────────────────┐
│ Prompt sent to LLM:                                     │
│                                                         │
│ <|im_start|>system                                      │
│ You are an AI agent, an assistance to the HPC Slurm admin│
│ <|im_end|>                                              │
│ <|im_start|>user                                        │
│ Reference Documents:                                    │
│ [Chunk 1] Compute Node... RAM epyc-jumbo 500GB...       │
│ [Chunk 2] Interactive Node... RAM 60GB...               │
│ [Chunk 4] System Software... Rocky Linux 8...           │
│ [Chunk 0] Intro ALELEON Mk.V...                        │
│ [Chunk 5] Package Manager... EasyBuild...               │
│                                                         │
│ Question: How much RAM in the epyc-jumbo partition?     │
│ <|im_end|>                                              │
│ <|im_start|>assistant                                   │
│                                                         │
│         ▼ Model generates token by token                │
│                                                         │
│ "The amount of RAM per node in the epyc-jumbo           │
│  partition is 500GB effective."                         │
└─────────────────────────────────────────────────────────┘
```

**Generation parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `temperature=0.1` | Very low → **deterministic** answers, not creative | Suitable for RAG (facts) |
| `top_p=0.9` | Nucleus sampling — only select tokens from the top 90% probability | Reduces random answers |
| `max_new_tokens=512` | Max 512 output tokens | Answer length limit |
| `max_model_len=4096` | Max 4096 total tokens (prompt + output) | Context window limit |
| `gpu_memory_utilization=0.90` | Use 90% VRAM | Save 10% for overhead |
| `enforce_eager=True` | Disable CUDAGraph | RDNA4 compatibility |

### RAG Chain — Combining Everything

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**`create_stuff_documents_chain`** — Strategy: **"Stuff"**

"Stuff" means: **put ALL chunks into 1 prompt at once**.

```
Other strategies (not used in this code):
┌─────────────────────────────────────────────────────────┐
│ Stuff     : All chunks → 1 prompt → 1 answer      ✅   │
│ Map-Reduce: Each chunk → answer → combine all          │
│ Refine    : Chunk 1 → answer → + Chunk 2 → refine      │
│ Map-Rerank: Each chunk → answer + score → pick best    │
└─────────────────────────────────────────────────────────┘
```

**`create_retrieval_chain`** combines the retriever + stuff chain:

```
User Input
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌─────────┐
│ Retriever│ ──→ │ Stuff Chain  │ ──→ │  Output  │
│ (Top-5)  │     │ (Prompt+LLM) │     │ {answer} │
└──────────┘     └──────────────┘     └─────────┘
    │                    │
    │ 5 relevant         │ Prompt with
    │ chunks             │ context + question
    ▼                    ▼
 From Chroma        To vLLM/GPU
```

---

## Full End-to-End Diagram

```
spesifikasi_aleleon.txt
        │
   [1] TextLoader
        │
        ▼
  1 Document (4076 chars)
        │
   [2] RecursiveCharacterTextSplitter
       (chunk_size=1000, overlap=200)
        │
        ▼
  7 Chunks (303, 778, 958, 3, 976, 907, 151 chars)
        │
   [3] all-MiniLM-L6-v2 (CPU, 22.7M params)
       Each chunk → 384-dimensional vector
        │
        ▼
   [4] Chroma DB (in-memory)
       7 vectors + 7 texts stored
        │
        │
   [5] vLLM + Qwen2.5-Coder-7B (GPU, 7B params)
       Model loaded into VRAM (14.37 GiB)
        │
        │
  ══════╪══════════════════════════════════════
  Per Question:
        │
  User: "How much RAM epyc-jumbo?"
        │
        ▼
  [a] Embed question → 384D vector (CPU)
        │
  [b] Cosine similarity vs 7 chunks in Chroma
        │
  [c] Retrieve top-5 most relevant chunks
        │
  [d] Insert into prompt template (ChatML)
        │
  [e] Send prompt to Qwen2.5 via vLLM (GPU)
        │
  [f] Model generates answer token-by-token
        │
        ▼
  "The amount of RAM per node in the
   epyc-jumbo partition is 500GB effective."
```