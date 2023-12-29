from flask import Flask, render_template, request, jsonify
import pickle
import os
import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import softmax
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from datetime import datetime
import json

import warnings
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
# Ignore all warnings
import openai
#from openai import OpenAI
api_key = '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'
#client = OpenAI(api_key=api_key)
openai.api_key = api_key
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load models and tokenizers
class_model_path = "checkpoint-200/"  
class_model = AutoModelForSequenceClassification.from_pretrained(class_model_path)
class_tokenizer = AutoTokenizer.from_pretrained(class_model_path)
blender_tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
blender_model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Download necessary NLTK data

mname = "facebook/blenderbot-400M-distill"
blender_model2 = BlenderbotForConditionalGeneration.from_pretrained(mname)
blender_tokenizer2 = BlenderbotTokenizer.from_pretrained(mname)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

api_key = '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'

analytics_data = {
    'total_queries': 0,
    'query_response_times': [],
    'error_count': 0,
    'query_types': {'chit-chat': 0, 'novel': 0},
    'query_frequencies': {},
    'unique_users': set(),
}

novels_data = {
    "sherlock_holmes": 0,
    "anthropology": 0,
    "england": 0,
    "peter": 0,
    "pride": 0,
    "proposal": 0,
    "romeo": 0,
    "salome": 0,
    "theodore": 0,
    "zorro": 0,
}

timeline_data = []

# Define helper functions
def blenderbot_response(input_text):
    inputs = blender_tokenizer([input_text], return_tensors='pt')
    reply_ids = blender_model.generate(**inputs)
    return blender_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

def blenderbot_response2(input_text):
    inputs = blender_tokenizer([input_text], return_tensors='pt', truncation=True, max_length=512)
    
    print("inputs", inputs)
    try:
        reply_ids = blender_model.generate(**inputs)
        response = blender_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    #    print("Blenderbot response:", response)
    except Exception as e:
        print("An error occurred:", e)
    #print("Tokenized input:", inputs)
    reply_ids = blender_model.generate(**inputs)
    return blender_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

def blenderbot_response3(input_text):
    inputs = blender_tokenizer2([input_text], return_tensors='pt', truncation=True, max_length=512)
    
    print("inputs", inputs)
    try:
        reply_ids = blender_model2.generate(**inputs)
        response = blender_tokenizer2.batch_decode(reply_ids, skip_special_tokens=True)[0]
    #    print("Blenderbot response:", response)
    except Exception as e:
        print("An error occurred:", e)
    #print("Tokenized input:", inputs)
    reply_ids = blender_model2.generate(**inputs)
    return blender_tokenizer2.batch_decode(reply_ids, skip_special_tokens=True)[0]


def classify_sentence(sentence):
    inputs = blender_tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = class_model(**inputs)
    probabilities = softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[0, predicted_class_id].item()
    return predicted_class_id, confidence

def classify_novels(nouns):
    novels = []
    k = ["sherlock_holmes.txt", "anthropology.csv", "england.txt", "peter.txt", "pride.csv", "proposal.csv", "romeo.csv", "salome.csv", "theodore.csv", "zorro.txt"]
    for filename in k:
        with open(filename, 'r') as f:
            novels.append(f.read().lower()) 
    vectorizer = TfidfVectorizer()
    document_term_matrix = vectorizer.fit_transform(novels)
    query_vector = vectorizer.transform([nouns])
    cosine_similarities = cosine_similarity(document_term_matrix, query_vector)
    most_similar_novel_index = cosine_similarities.argmax()
    return k[most_similar_novel_index]

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

book_to_button = {
    "all": 0,
    "sherlock_holmes": 1,
    "anthropology": 2,
    "england": 3,
    "peter": 4,
    "pride": 5,
    "proposal": 6,
    "romeo": 7,
    "salome": 8,
    "theodore": 9,
    "zorro": 10,
}

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    start_time = datetime.now()
    data = request.get_json()
    user_query = data['user_input']
    user_id = data.get('user_id', 'unknown')
    print("user_query",user_query)
    analytics_data['unique_users'].add(user_id)
    words = word_tokenize(user_query)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    nouns = ' '.join([word for word, tag in pos_tag(filtered_words) if tag.startswith('NN')])
    
    #blender_response = blenderbot_response2(user_query)
    blender_response = user_query
    print("blender_response",blender_response)
    predicted_class, confidence = classify_sentence(blender_response)
    '''
    if confidence > 0.55:
        print("predicted_class: novel")
    else:
        print("predicted_class: chit-chat")
    print("predicted_class",predicted_class)
    '''
    print("confidence",confidence)
    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()
    analytics_data['total_queries'] += 1
    analytics_data['query_response_times'].append(response_time)

    # Track query type
    query_type = 'chit-chat' if predicted_class == 0 else 'novel'
    analytics_data['query_types'][query_type] += 1

    # Track query frequency
    analytics_data['query_frequencies'][user_query] = analytics_data['query_frequencies'].get(user_query, 0) + 1

    if predicted_class == 0 and confidence > 0.55 : #Its a chit-chat
        response = blenderbot_response2(user_query)
        button_id = 0
        # timeline_data.append([str(analytics_data["total_queries"]), "Queries", 0])
        timeline_data.append([str(analytics_data["total_queries"]), "Queries", 0, "chit-chat"])
        
    #elif predicted_class == 1 and confidence > 0.54:
    elif predicted_class == 1 and confidence > 0.52: #Its a novel
        #response = blender_response
        novels = []
        book_name = classify_novels(str(nouns))
        query = user_query
        filename_without_extension = os.path.splitext(book_name)[0]
        print(filename_without_extension)
        bookname = filename_without_extension
        button_id = book_to_button.get(bookname, None)
        print("button_id",button_id)
        with open('./database_store.pkl', 'rb') as f:
            store = pickle.load(f)

        from index_documents import get_answer
        print("query",query)
        print("bookname",bookname)
        novels_data[bookname] += 1
        #answer = get_answer(query, bookname)
        top_k_paragraphs = 4
        answer = get_answer(query, bookname, top_k=top_k_paragraphs)
        question = answer[0]
        context = answer[1]
        context_str = [q['metadata']['document_id'] + "_" + str(int(q['metadata']['sentence_id'])) for q in context]
        context = [store[x] for x in context_str]
        print("TOP k",question_and_answer_prompt(question, context))
        print(question_and_answer_prompt(question, context))
        #response = blenderbot_response2(question_and_answer_prompt(question, context))
        response_id =  openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt= question_and_answer_prompt(question, context) , max_tokens=100)
        response = response_id.choices[0].text.strip()
        #response = blenderbot_response2(question_and_answer_prompt(question, context)) 
        print("response",response)

        timeline_data.append([str(analytics_data["total_queries"]), "Queries", book_to_button[bookname], bookname])

    else:
        response = blenderbot_response2(user_query)
        button_id = 0
        # timeline_data.append(json.dumps({"x": str(analytics_data["total_queries"]), "y": "Queries", "heat": 1}))
        timeline_data.append([str(analytics_data["total_queries"]), "Queries", 0, "chit-chat"])
    

    #print("response",response)
    return jsonify({"response": response, "button_id": button_id})
    #return jsonify({"response": response})

@app.route("/dashboard")
def dashboard():
    #return "Dashboard route is working"
    avg_response_time = sum(analytics_data['query_response_times']) / len(analytics_data['query_response_times']) if analytics_data['query_response_times'] else 0
    max_response_time = max(analytics_data['query_response_times'], default=0)
    min_response_time = min(analytics_data['query_response_times'], default=0)
    error_rate = (analytics_data['error_count'] / analytics_data['total_queries']) * 100 if analytics_data['total_queries'] else 0
    most_frequent_query = max(analytics_data['query_frequencies'], key=analytics_data['query_frequencies'].get, default="None")
    unique_users_count = len(analytics_data['unique_users'])

    # json_timeline = json.dumps(timeline_data)
    print(timeline_data)

    return render_template('dashboard.html', 
                           total_queries=analytics_data['total_queries'],
                           avg_response_time=avg_response_time,
                           max_response_time=max_response_time,
                           min_response_time=min_response_time,
                           query_types=analytics_data['query_types'],
                           error_rate=error_rate,
                           most_frequent_query=most_frequent_query,
                           unique_users_count=unique_users_count,
                           novels_data=novels_data,
                           timeline_data=timeline_data)

if __name__ == "__main__":
    app.run(port=5001, host="0.0.0.0")
