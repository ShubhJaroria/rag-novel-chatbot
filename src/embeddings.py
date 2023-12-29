# backend/rag/vectordb.py
import pinecone
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from fastapi import HTTPException
import os
import time

TOP_K_DOCUMENTS = 50
INDEX_NAME = 'indexer'


openapi_config = OpenApiConfiguration.get_default_copy()
pinecone.init(
    api_key='17151802-3575-4559-b516-1a07a5bb22d2', 
    environment="gcp-starter",
    openapi_config=openapi_config)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def add_document_to_db(index_name, document_id: str, paragraphs: list[str], embeddings: list[str]):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384
                              , metadata_config={"indexed": ["document_id"]})

    index = pinecone.Index(index_name)

    try:
        embeddings = [
            (
            f"{document_id}_{i}", # Id of vector
            embedding ,  # Dense vector values
            {"document_id": document_id, "sentence_id": i} 
            # For ease of architecture I will save the text in pinecone as well
            # This is not recommended since Pinecone memory might be expensive
            )
            for i, (embedding) in enumerate(zip(embeddings)) 
        ]
        #print(len(embeddings))
        for embedding_chunk in chunks(embeddings, 100):
            index.upsert(
                vectors=embedding_chunk
            )
            #time.sleep(5)
                
    except Exception as e:
        raise HTTPException(404, detail= f'Pinecone indexing fetch fail with error {e}')
'''
def fetch_top_paragraphs(index_name, document_id: str, embedding: list[float]) -> list[str]:
    index = pinecone.Index(index_name)
    try:
        query_response = index.query(
            top_k=TOP_K_DOCUMENTS,
            vector=embedding,
            filter={
                "document_id": {"$eq": document_id},
            },
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(404, detail= f'Pinecone indexing fetch fail with error {e}')
    
    return query_response['matches']
'''
def fetch_top_paragraphs(index_name, document_id: str, embedding: list[float], top_k=5) -> list[str]:
    index = pinecone.Index(index_name)
    try:
        query_response = index.query(
            top_k=top_k,  # Use the passed top_k parameter
            vector=embedding,
            filter={"document_id": {"$eq": document_id}},
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(404, detail= f'Pinecone indexing fetch fail with error {e}')
    
    return query_response['matches']

