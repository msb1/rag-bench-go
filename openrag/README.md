---
license: cc-by-nc-4.0
---
# Open RAG Benchmark

The Open RAG Benchmark is a unique, high-quality Retrieval-Augmented Generation (RAG) dataset constructed directly from arXiv PDF documents, specifically designed for evaluating RAG systems with a focus on multimodal PDF understanding. Unlike other datasets, Open RAG Benchmark emphasizes **pure PDF content**, meticulously extracting and generating queries on diverse modalities including **text, tables, and images**, even when they are intricately interwoven within a document.

This dataset is purpose-built to power the company's [Open RAG Evaluation project](https://github.com/vectara/open-rag-eval), facilitating a holistic, end-to-end evaluation of RAG systems by offering:

  - **Richer Multimodal Content:** A corpus derived exclusively from PDF documents, ensuring fidelity to real-world data and encompassing a wide spectrum of text, tabular, and visual information, often with intermodal crossovers.
  - **Tailored for Open RAG Evaluation:** Designed to support the unique and comprehensive evaluation metrics adopted by the Open RAG Evaluation project, enabling a deeper understanding of RAG performance beyond traditional metrics.
  - **High-Quality Retrieval Queries & Answers:** Each piece of extracted content is paired with expertly crafted retrieval queries and corresponding answers, optimized for robust RAG training and evaluation.
  - **Diverse Knowledge Domains:** Content spanning various scientific and technical domains from arXiv, ensuring broad applicability and challenging RAG systems across different knowledge areas.

The current draft version of the Arxiv dataset, as the first step in this multimodal RAG dataset collection, includes:

  - **Documents:** 1000 PDF papers evenly distributed across all Arxiv categories.
      - 400 positive documents (each serving as the golden document for some queries).
      - 600 hard negative documents (completely irrelevant to all queries).
  - **Multimodal Content:** Extracted text, tables, and images from research papers.
  - **QA Pairs:** 3045 valid question-answer pairs.
      - **Based on query types:**
          - 1793 abstractive queries (requiring generating a summary or rephrased response using understanding and synthesis).
          - 1252 extractive queries (seeking concise, fact-based answers directly extracted from a given text).
      - **Based on generation sources:**
          - 1914 text-only queries
          - 763 text-image queries
          - 148 text-table queries
          - 220 text-table-image queries

## Dataset Structure

The dataset is organized similar to the [BEIR dataset](https://github.com/beir-cellar/beir) format within the `official/pdf/arxiv/` directory.

```
official/
└── pdf
    └── arxiv
        ├── answers.json
        ├── corpus
        │   ├── {PAPER_ID_1}.json
        │   ├── {PAPER_ID_2}.json
        │   └── ...
        ├── pdf_urls.json
        ├── qrels.json
        └── queries.json
```

Each file's format is detailed below:

### `pdf_urls.json`

This file provides the original PDF links to the papers in this dataset for downloading purposes.

```json
{
    "Paper ID": "Paper URL",
    ...
}
```

### `corpus/`

This folder contains all processed papers in JSON format.

```json
{
    "title": "Paper Title",
    "sections": [
        {
            "text": "Section text content with placeholders for tables/images",
            "tables": {"table_id1": "markdown_table_string", ...},
            "images": {"image_id1": "base64_encoded_string", ...},
        },
        ...
    ],
    "id": "Paper ID",
    "authors": ["Author1", "Author2", ...],
    "categories": ["Category1", "Category2", ...],
    "abstract": "Abstract text",
    "updated": "Updated date",
    "published": "Published date"
}
```

### `queries.json`

This file contains all generated queries.

```json
{
    "Query UUID": {
        "query": "Query text",
        "type": "Query type (abstractive/extractive)",
        "source": "Generation source (text/text-image/text-table/text-table-image)"
    },
    ...
}
```

### `qrels.json`

This file contains the query-document-section relevance labels.

```json
{
    "Query UUID": {
        "doc_id": "Paper ID",
        "section_id": Section Index
    },
    ...
}
```

### `answers.json`

This file contains the answers for the generated queries.

```json
{
    "Query UUID": "Answer text",
    ...
}
```

## Dataset Creation

The Open RAG Benchmark dataset is created through a systematic process involving document collection, processing, content segmentation, query generation, and quality filtering.

1.  **Document Collection:** Gathering documents from sources like Arxiv.
2.  **Document Processing:** Parsing PDFs via OCR into text, Markdown tables, and base64 encoded images.
3.  **Content Segmentation:** Dividing documents into sections based on structural elements.
4.  **Query Generation:** Using LLMs (currently `gpt-4o-mini`) to generate retrieval queries for each section, handling multimodal content such as tables and images.
5.  **Quality Filtering:** Removing non-retrieval queries and ensuring quality through post-processing via a set of encoders for retrieval filtering and `gpt-4o-mini` for query quality filtering.
6.  **Hard-Negative Document Mining (Optional):** Mining hard negative documents that are entirely irrelevant to any existing query, relying on agreement across multiple embedding models for accuracy.

The code for reproducing and customizing the dataset generation process is available in the [Open RAG Benchmark GitHub repository](https://www.google.com/search?q=https://github.com/vectara/Open-RAG-Benchmark).

## Limitations and Challenges

Several challenges are inherent in the current dataset development process:

  - **OCR Performance:** Mistral OCR, while performing well for structured documents, struggles with unstructured PDFs, impacting the quality of extracted content.
  - **Multimodal Integration:** Ensuring proper extraction and seamless integration of tables and images with corresponding text remains a complex challenge.

## Future Enhancements

The project aims for continuous improvement and expansion of the dataset, with key next steps including:

### Enhanced Dataset Structure and Usability:

  - **Dataset Format and Content Enhancements:**
      - **Rich Metadata:** Adding comprehensive document metadata (authors, publication date, categories, etc.) to enable better filtering and contextualization.
      - **Flexible Chunking:** Providing multiple content granularity levels (sections, paragraphs, sentences) to accommodate different retrieval strategies.
      - **Query Metadata:** Classifying queries by type (factual, conceptual, analytical), difficulty level, and whether they require multimodal understanding.
  - **Advanced Multimodal Representation:**
      - **Improved Image Integration:** Replacing basic placeholders with structured image objects including captions, alt text, and direct access URLs.
      - **Structured Table Format:** Providing both markdown and programmatically accessible structured formats for tables (headers/rows).
      - **Positional Context:** Maintaining clear positional relationships between text and visual elements.
  - **Sophisticated Query Generation:**
      - **Multi-stage Generation Pipeline:** Implementing targeted generation for different query types (factual, conceptual, multimodal).
      - **Diversity Controls:** Ensuring coverage of different difficulty levels and reasoning requirements.
      - **Specialized Multimodal Queries:** Generating queries specifically designed to test table and image understanding.
  - **Practitioner-Focused Tools:**
      - **Framework Integration Examples:** Providing code samples showing dataset integration with popular RAG frameworks (LangChain, LlamaIndex, etc.).
      - **Evaluation Utilities:** Developing standardized tools to benchmark RAG system performance using this dataset.
      - **Interactive Explorer:** Creating a simple visualization tool to browse and understand dataset contents.

### Dataset Expansion:

  - Implementing alternative solutions for PDF table & image extraction.
  - Enhancing OCR capabilities for unstructured documents.
  - Broadening scope beyond academic papers to include other document types.
  - Potentially adding multilingual support.

## Acknowledgments

The Open RAG Benchmark project uses OpenAI's GPT models (specifically `gpt-4o-mini`) for query generation and evaluation. For post-filtering and retrieval filtering, the following embedding models, recognized for their outstanding performance on the [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard), were utilized:

  - [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)
  - [dunzhang/stella\_en\_1.5B\_v5](https://huggingface.co/NovaSearch/stella_en_1.5B_v5)
  - [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
  - [infly/inf-retriever-v1](https://huggingface.co/infly/inf-retriever-v1)
  - [Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral)
  - [openai/text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large)