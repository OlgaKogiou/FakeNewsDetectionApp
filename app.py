"""
Author: Olga Kogiou
MACHINE LEARNING WEB APP USING STREAMLIT
A simple web app which uses:
- pretrained fixed models of a Kaggle dataset of Fake and True News
- Streamlit GUI and Virtual Environment
- Pipfile of all requirements
- Dockfile for Docker deployment
- .yml file as a simple stack file to run and manage our app in a Docker Swarm

Our goal is to build a web app that implements all main features from our dataset. After, we deploy the app to a docker
container that contains an image of our app. Lastly, docker swarm is implemented in order to prevent damage and have
better speed.

By using the URL mode you can paste the URL you want and the text will be used automatically. By using the Paste mode
you can copy-paste the article yourself.

- For prediction: we use pre-trained models stored in .pickle files and a Vectorizer also stored. We apply the Vectorizer
and then wait for the prediction result from our models.
- For summarization: we use Summy Summarization
- For NLP: we make both a wordcloud which contains all the most common words in the article and a tablr with main
features such as Lemma, Part of speech tags and Token (for each word).
- For NER checker: we use spacy's display to make all entities visual.
"""

# libraries import
from gensim.summarization import summarize
import streamlit as st
import joblib
import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from spacy import displacy
from PIL import Image
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen
nltk.download('punkt')
matplotlib.use("Agg")
nlp = spacy.load('en_core_web_sm')

# load Vectorizer for the article
news_vectorizer = open("models/Vectorizer_cv.pickle", "rb")
news_cv = joblib.load(news_vectorizer)

# To be able to render our NER in spacy we will be using displacy and wrap our result within an html as below.
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


# load our fixed pre-trained models
def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

# Named Entity Recognition of the article
def NER_checker(raw_text):
    st.subheader("Named Entity Recognition with Spacy")
    if st.button("Analyze"):
        docx = analyze_text(raw_text)
        html = displacy.render(docx, style="ent")
        html = html.replace("\n\n", "\n")
        # HTML WRAPPER
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        # Analysis of the entities of the article
        result = [(entity.text, entity.label_) for entity in docx.ents]
        st.json(result)


# Prediction
def prediction(news_text):
    st.info("Prediction with ML")
    # choices for predictions
    all_ml_models = ["Random Forest", "Naive Bayes", "Support Vector Machine"]
    model_choice = st.selectbox("Select Model", all_ml_models)
    # labels of prediction
    prediction_labels = {'Fake': '1', 'Real': '0'}
    if st.button("Classify"):
        # Classification action
        st.text("Original Text::\n{}".format(news_text))
        # use of the pretrained model
        vect_text = news_cv.transform([news_text]).toarray()
        # enery classifier has its own model for prediction
        if model_choice == 'Random Forest':
            predictor = load_prediction_models("models/RandomForest.pickle")
            prediction = predictor.predict(vect_text)
        elif model_choice == 'Naive Bayes':
            predictor = load_prediction_models("models/NaiveBayes.pickle")
            prediction = predictor.predict(vect_text)
        elif model_choice == 'Support Vector Machine':
            predictor = load_prediction_models("models/SupportVectorMachine.pickle")
            prediction = predictor.predict(vect_text)
            # dictionary that matches label(Real Fake) to the value(1,0)
        final_result = get_key(prediction, prediction_labels) 
        st.success("News Categorized as:: {}".format(final_result))


# Wrapper Function for Tokens, Part-of-Speech Tags, Lemma
def Tabulize(raw_text):
    docx = nlp(raw_text)
    c_tokens = [token.text for token in docx]
    c_lemma = [token.lemma_ for token in docx]
    c_pos = [token.pos_ for token in docx]
    # Creation of a pandas dataframe to print to user
    new_df = pd.DataFrame(zip(c_tokens, c_lemma, c_pos), columns=['Tokens', 'Lemma', 'POS'])
    st.dataframe(new_df)


# Function for word cloud picture
def wordcloud(raw_text):
    c_text = raw_text
    wordcloud = WordCloud().generate(c_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# Function for Natural Language Processing
def NLP(raw_text):
    st.info("Natural Language Processing of Text")
    nlp_task = ["Tokenization", "Lemmatization", "POS Tags"]
    # user selection
    task_choice = st.selectbox("Choose NLP Task", nlp_task)
    # Button action
    if st.button("Tabulize"):
        Tabulize(raw_text)
    # Check box
    if st.checkbox("WordCloud"):
        wordcloud(raw_text)


# Get the Keys (Fake:1, True:0)
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# Fetch Text From Url
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text,
                                soup.find_all('p')))
    return fetched_text


# Analyse text using SpaCy
def analyze_text(text):
    return nlp(text)


# Function for Summarization
def Summarize(raw_text):
    st.subheader("Summarize Document")
    summarizer_type = st.selectbox("Summarizer Type", ["Gensim", "Sumy Lex Rank"])
    # summarization function
    if st.button("Summarize"):
        if summarizer_type == "Gensim":
            summary_result = summarize(raw_text)
        elif summarizer_type == "Sumy Lex Rank":
            summary_result = sumy_summarizer(raw_text)
        # print result to user
        st.write(summary_result)


def main():
    st.title("News Classifier")
    # html code for front page markdown
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h1 style="color:white;text-align:center;">Fake News Detector </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # the two modes
    text_choices = ["Paste text", "URL"]
    text_choice = st.selectbox("Select Choice", text_choices)
    news_text = ''
    if text_choice == "Paste text":
        news_text = st.text_area("Enter News Here", "Type Here")
    elif text_choice == "URL":
        raw_url = st.text_input("Enter URL Here", "Type here")
        # error catching for url paste
        if raw_url != "Type here":
            if raw_url == "" or raw_url == '"':
                st.write("Waiting...")
            else:
                news_text = get_text(raw_url)
    # main modes for user
    activity = ['Prediction', 'NLP', 'Summarize', 'NER Checker']
    choice = st.sidebar.selectbox("Select Activity", activity)
    # function calls
    if choice == 'Prediction':
        prediction(news_text)
    elif choice == 'NLP':
        NLP(news_text)
    elif choice == 'Summarize':
        Summarize(news_text)
    elif choice == 'NER Checker':
        NER_checker(news_text)
    
    st.sidebar.subheader("About")

# call of main
if __name__ == '__main__':
    main()
