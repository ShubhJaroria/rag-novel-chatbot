from index_documents import add_document
import os

dir_list = os.listdir("./novels/")

for book in dir_list:
    #print(book, book[-4:])
    if book[-4:] != ".txt": continue
    add_document("./novels/" + book, book[:-4].lower())



