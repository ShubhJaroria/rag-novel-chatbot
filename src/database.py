from nltk.tokenize import sent_tokenize

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

    return make_paragraph(sentences, 4)

def read_document_from_file(filepath):
    with open(filepath, 'r') as f:
        return '\n'.join(f.readlines())

# Add document to the database and return id of the document
def add_document(filepath: str, bookname) -> str:
    document_text = read_document_from_file(filepath)
    paragraphs = split_document_to_paragraphs(document_text)
    if len(paragraphs) == 0:
        raise HTTPException('404', detail='No text was extracted from the document')
    document_id = bookname
    add_document_to_store("indexer", document_id, paragraphs)
    print(document_id)
    return document_id


store = {}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def add_document_to_store(index_name, document_id: str, paragraphs: list[str]):
    for i in range(len(paragraphs)):
        store[document_id + "_" + str(i)] = paragraphs[i]    


import os

dir_list = os.listdir("./novels/")

for book in dir_list:
    if book[-4:] != ".txt": continue
    add_document("./novels/" + book, book[:-4].lower())      
    

import pickle 

with open('./database_store.pkl', 'wb') as f:
    pickle.dump(store, f)      