# RAG (Retrieval-Augmented Generation) Application

A document processing and querying system built with LangChain and Amazon Bedrock that enables semantic search across PDF documents.

## Overview

This RAG application consists of three main components:

1. Document processing and database creation (`create_database.py`)
2. Embedding generation using Amazon Bedrock (`get_embedding_functions.py`) 
3. Query interface with semantic search (`query_database.py`)

## Prerequisites

- Python 3.x
- AWS Account with Bedrock access
- AWS CLI configured with credentials
- Required Python packages:
  - langchain
  - langchain_community 
  - langchain_aws
  - boto3
  - chromadb

## Installation

Clone the repository

    git clone https://github.com/yourusername/rag-application.git
    cd rag-application

    
Install dependencies

    
    pip install -r requirements.txt

    

Configure AWS credentials

    
    aws configure

    

## Usage

 - Create a data/ dir and add PDF documents  data/ directory

- Create/update the database:

    
      python create_database.py

    

- Reset the database if needed:

    
      python create_database.py --reset


- Run queries against the processed documents:

    
      python query_database.py "your question here"

    

## Technical Details
- Document Processing:
    - Uses PyPDFDirectoryLoader for PDF processing
    - Chunks documents into 1000 character segments with 500 character overlap
    - Generates unique chunk IDs in format: source:page:chunk_index
    - Stores vectors in Chroma database

- Embeddings:
    - Uses Amazon Bedrock Titan model for embeddings
    - Configured with:
        - Max retries: 3
        - Connection timeout: 30s
        - Read timeout: 30s

- Query Processing:
  - Performs similarity search to find top 3 relevant chunks
  - Uses Amazon Bedrock Nova-Lite model for response generation
  - Includes source attribution in responses


