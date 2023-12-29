query = "Who's sherlock?"
bookname = "sherlock_holmes"

import pickle

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

with open('./database_store.pkl', 'rb') as f:
    store = pickle.load(f)

from index_documents import get_answer
answer = get_answer(query, bookname)

question = answer[0]
context = answer[1]

context_str = [q['metadata']['document_id'] + "_" + str(int(q['metadata']['sentence_id'])) for q in context]
context = [store[x] for x in context_str]

print(question_and_answer_prompt(question, context))
