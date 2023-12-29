from nltk.tokenize import sent_tokenize
import uuid
from rag import fetch_embeddings, question_and_answer_prompt
from embeddings import add_document_to_db, fetch_top_paragraphs
from fastapi import HTTPException

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def make_paragraph(lst, n):
    i = 0
    tmp = ""
    ans = []
    while(i < len(lst)):
        tmp = tmp + lst[i] + "/n"
        i += 1
        print(tmp)
        if i%n == 0:
            ans.append(tmp)
            tmp = ""

    ans.append(tmp)        

    return ans

# Split document into paragraph of roughly of max_char size
def split_document_to_paragraphs(document: str, paragraph_len: int = 100) -> list[str]:
    sentences = sent_tokenize(document) # Split the paragraph by sentences
    
    return make_paragraph(sentences,4)

def read_document_from_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read().decode('utf-8')

# Add document to the database and return id of the document
def add_document(filepath: str, bookname) -> str:
    document_text = read_document_from_file(filepath)
    #print(document_text)
    paragraphs = split_document_to_paragraphs(document_text)
    if len(paragraphs) == 0:
        raise HTTPException('404', detail='No text was extracted from the document')
    embeddings = fetch_embeddings(paragraphs, embedding_type='search_document')
    #print(len(paragraphs))
    #print(len(embeddings))

    document_id = bookname
    add_document_to_db("indexer", document_id, paragraphs, embeddings)
    print(document_id)
    return document_id
'''
def get_answer(question: str, bookname: str):
    embedding = fetch_embeddings([question], embedding_type='search_query')[0]
    relevant_paragraphs = fetch_top_paragraphs("indexer", bookname, embedding)
    if len(relevant_paragraphs) == 0:
        raise HTTPException(404, detail='Embedding are not ready yet for this document')
    return (question, relevant_paragraphs)
'''

def get_answer(question: str, bookname: str, top_k=5):
    embedding = fetch_embeddings([question], embedding_type='search_query')[0]
    relevant_paragraphs = fetch_top_paragraphs("indexer", bookname, embedding, top_k)
    if len(relevant_paragraphs) == 0:
        raise HTTPException(404, detail='Embedding are not ready yet for this document')
    return (question, relevant_paragraphs)





    