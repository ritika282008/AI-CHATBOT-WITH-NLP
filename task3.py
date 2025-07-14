import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

corpus = """
Hello! How can I help you today?
I am a chatbot created to assist with your questions.
I can help you with weather, facts, and more.
How are you feeling today?
Tell me about your day.
What can I do for you?
Bye! Have a great day!
"""

sent_tokens = nltk.sent_tokenize(corpus.lower())  
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_input):
    sent_tokens.append(user_input.lower())
    tfidf_vec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidf_vec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    score = flat[-1]

    if score == 0:
        return "I’m sorry, I don’t understand."
    else:
        return sent_tokens[idx]

print("AI Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("AI Chatbot: Goodbye! 👋")
        break
    else:
        print("AI Chatbot:", response(user_input))
        sent_tokens.pop()  
