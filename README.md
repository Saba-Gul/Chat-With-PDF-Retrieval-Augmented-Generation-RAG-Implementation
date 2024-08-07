# Chat With PDF: Retrieval-Augmented Generation (RAG) Implementation

This repository contains code that demonstrates how to interact with a PDF document using Retrieval-Augmented Generation (RAG). The code loads a PDF, processes it into smaller chunks, stores the embeddings in a vector store, and performs similarity searches to retrieve relevant document chunks based on a query. It then formats the context and question into a prompt and generates a response using OpenAI's language model.

## Overview

The goal of this implementation is to handle large PDF documents efficiently and provide accurate answers based on the content of the PDF. The main steps involved are:

1. **Loading and Ingesting Data**: Load a PDF document and split it into smaller chunks.
2. **Preprocessing**: Tokenize and preprocess the text for embedding generation.
3. **Vector Embeddings**: Create and store vector embeddings for document chunks.
4. **Similarity Search**: Perform similarity searches to find relevant document chunks.
5. **RAG Enriched Prompt**: Create prompts with retrieved context and generate answers using an LLM (Language Learning Model).

![image](https://i0.wp.com/www.phdata.io/wp-content/uploads/2023/11/image1-3.png)
*Credit: [phdata](https://www.phdata.io/blog/what-is-retrieval-augmented-generation-rag/)*

## Setup

To get started, ensure you have the necessary dependencies installed. You can install them using the following commands:

```bash
pip install openai langchain chromadb tiktoken
pip install -U langchain-community
pip install pypdf
```

## Code Walkthrough

### Load and Ingest Data

Load a PDF document using `PyPDFLoader` and split it into smaller chunks.

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('/content/attentionisallyouneed.pdf')
pages = loader.load()
```

### Preprocessing

Split the document into smaller chunks and tokenize the text.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

encoding = tiktoken.encoding_for_model("text-embedding-3-small")
doc_tokens = [len(encoding.encode(page.page_content)) for page in docs]
total_tokens = sum(doc_tokens)
cost = total_tokens * 0.0004
print(f"Total tokens: {total_tokens}")
print(f"Cost: ${cost}")
```

### Vector Embeddings

Create vector embeddings for the document chunks and store them using Chroma.

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai.api_key)
db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory='my-embeddings')
```

### Similarity Search

Perform a similarity search to retrieve relevant document chunks based on a query.

```python
results = db.similarity_search_with_relevance_scores('What are self attention?', k=5)

for (doc, score) in results:
    print('score', score)
    print(doc)
    print('-------------------')
```

### RAG Enriched Prompt

Create a prompt with the retrieved context and generate an answer using OpenAI's language model.

```python
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI

question = 'Please give me an introduction to transformer architecture'
context_docs = db.similarity_search(question, k=5)

prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say don't know. Do not try to make up the answer.

    <context>
    {context}
    </context>

    Question: {question}
    Helpful Answer""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.9, api_key=openai.api_key)
qa_chain = LLMChain(llm=llm, prompt=prompt)

result = qa_chain({
    'question': question,
    'context': "\n".join([doc.page_content for doc in context_docs])
})

print(result)
```

## Credits

This implementation is inspired by the blog post on Retrieval-Augmented Generation (RAG) by [phData](https://www.phdata.io/blog/what-is-retrieval-augmented-generation-rag/).
