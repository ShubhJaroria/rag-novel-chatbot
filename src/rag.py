import os
import time
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose different pre-trained models


def fetch_embeddings(texts: list[str], embedding_type: str = 'search_document') -> list[list[float]]:
    embeddings = model.encode(texts, convert_to_tensor=False).tolist()  # Use convert_to_tensor=True for GPU acceleration
    #print(len(texts), len(embeddings), len(embeddings[0]))
    return embeddings
    

def question_and_answer_prompt(question: str, context: list[str]) -> str:
    context_str = '\n'.join(context)
    return f"""
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer: 
    """
