# Nutrition RAG

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python and several open-source libraries. The goal of this project is to answer nutrition-related questions based on a given set of documents.

The implementation is provided in the `RAG2.ipynb` Jupyter Notebook.

## Table of Contents

- [Introduction to RAG](#introduction-to-rag)
- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
  - [Dependencies](#dependencies)
  - [Hugging Face API Token](#hugging-face-api-token)
- [Implementation Steps](#implementation-steps)
  - [Step 1: Loading the Data](#step-1-loading-the-data)
  - [Step 2: Splitting Documents into Chunks](#step-2-splitting-documents-into-chunks)
  - [Step 3: Generating Text Embeddings](#step-3-generating-text-embeddings)
  - [Step 4: Creating a Vector Store with FAISS](#step-4-creating-a-vector-store-with-faiss)
  - [Step 5: Setting Up the Large Language Model (LLM)](#step-5-setting-up-the-large-language-model-llm)
  - [Step 6: Creating the RetrievalQA Chain](#step-6-creating-the-retrievalqa-chain)
  - [Step 7: Running a Query](#step-7-running-a-query)
- [Example Usage](#example-usage)


## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models (LLMs) with the ability to retrieve relevant information from a custom knowledge base. This approach allows the LLM to generate more accurate and context-aware responses, especially for domain-specific tasks.

In this project, we use a RAG pipeline to build a question-answering system for nutrition.

## Pipeline Overview

The RAG pipeline in this project consists of the following steps:

1.  **Load Documents:** Load text documents from a specified directory.
2.  **Split Text:** Split the documents into smaller, manageable chunks.
3.  **Generate Embeddings:** Convert the text chunks into numerical vectors (embeddings) using a sentence transformer model.
4.  **Create Vector Store:** Store the embeddings in a vector store (FAISS) for efficient similarity search.
5.  **Retrieve Relevant Documents:** Given a user's query, retrieve the most relevant document chunks from the vector store.
6.  **Generate Answer:** Pass the retrieved documents and the user's query to a large language model (LLM) to generate a human-like answer.

## Setup

### Dependencies

First, you need to install the required Python libraries. You can install them using `pip`:

```bash
pip install langchain unstructured faiss-cpu sentence-transformers
```

You will also need to install the following libraries for specific document types:

```bash
pip install pypdf tiktoken
```

### Hugging Face API Token

This project uses a large language model from the Hugging Face Hub. To use it, you need to have a Hugging Face Hub API token.

1.  If you don't have an account, create one on the [Hugging Face website](https://huggingface.co/join).
2.  Get your API token from your Hugging Face account settings.
3.  Set your API token as an environment variable named `HUGGINGFACEHUB_API_TOKEN`.

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_TOKEN"
```

## Implementation Steps

The following sections explain each step of the RAG pipeline as implemented in the `RAG2.ipynb` notebook.

### Step 1: Loading the Data

**Explanation:**
The first step is to load the text documents that will serve as the knowledge base for our RAG pipeline. We use the `DirectoryLoader` from the `langchain` library to load all `.txt` files from the `data/` directory.

**Code:**
```python
from langchain.document_loaders import DirectoryLoader, TextLoader

# Define the path to your data directory
DATA_PATH = 'data/'

# Load the documents
loader = DirectoryLoader(DATA_PATH, glob='*.txt', loader_cls=TextLoader)
documents = loader.load()
```

### Step 2: Splitting Documents into Chunks

**Explanation:**
Large documents are difficult to process at once. Therefore, we split them into smaller, overlapping chunks. This allows the model to process smaller pieces of text, which is more efficient. We use the `RecursiveCharacterTextSplitter` for this purpose.

**Code:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split the documents into chunks
texts = text_splitter.split_documents(documents)
```

### Step 3: Generating Text Embeddings

**Explanation:**
To find relevant documents for a given query, we need to represent the text chunks in a numerical format. We use embeddings for this. We use the `HuggingFaceEmbeddings` class to load the `sentence-transformers/all-MiniLM-L6-v2` model, which will generate a vector for each text chunk.

**Code:**
```python
from langchain.embeddings import HuggingFaceEmbeddings

# Define the embedding model
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Create the embeddings object
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
```

### Step 4: Creating a Vector Store with FAISS

**Explanation:**
A vector store is used to store the embeddings and perform efficient similarity searches. We use `FAISS` (Facebook AI Similarity Search) to create a vector store from our text chunks and their corresponding embeddings. This will allow us to quickly find the most relevant text chunks for a given query.

**Code:**
```python
from langchain.vectorstores import FAISS

# Create the vector store from the text chunks and embeddings
db = FAISS.from_documents(texts, embeddings)
```

### Step 5: Setting Up the Large Language Model (LLM)

**Explanation:**
The "generation" part of our RAG pipeline is handled by a large language model. We use the `databricks/dolly-v2-3b` model from the Hugging Face Hub. We initialize the model using the `HuggingFaceHub` class from `langchain`.

**Code:**
```python
from langchain import HuggingFaceHub

# Initialize the language model
llm = HuggingFaceHub(
    repo_id="databricks/dolly-v2-3b",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)
```

### Step 6: Creating the RetrievalQA Chain

**Explanation:**
The `RetrievalQA` chain ties everything together. It takes a retriever (our FAISS vector store) and an LLM, and creates a pipeline for question-answering. The `stuff` chain type means that all retrieved documents are "stuffed" into the prompt that is sent to the LLM.

**Code:**
```python
from langchain.chains import RetrievalQA

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2})
)
```

### Step 7: Running a Query

**Explanation:**
Finally, we can ask a question by calling the `run` method of our `qa_chain`. The chain will perform the following actions:
1.  Take the query and find the most relevant document chunks from the FAISS vector store.
2.  Pass the retrieved chunks and the query to the `dolly-v2-3b` model.
3.  Return the model's generated answer.

**Code:**
```python
query = "What are the benefits of a high-protein diet?"
result = qa_chain.run(query)
print(result)
```

## Example Usage

Let's say your documents contain information about the benefits of a high-protein diet.

**Query:**

```
What are the benefits of a high-protein diet?
```

**Expected Answer (generated by the RAG pipeline):**

```
A high-protein diet can help with weight loss by increasing satiety and metabolism. It can also help build and maintain muscle mass, especially when combined with resistance training. Additionally, protein is essential for various bodily functions, including hormone production and immune system support.
```
