# FakeNewsDetectionWebApp
# Streamlit App
## In order to run the App you have to download the models folder: https://drive.google.com/drive/folders/1-6mC3z23EQnuw8oWdoMXvGHUHymVggMS?usp=sharing and put it in the same folder with the rest of the files.

### Author: Olga Kogiou
### MACHINE LEARNING WEB APP USING STREAMLIT
A simple web app which uses:
- pretrained fixed models of a Kaggle dataset of Fake and True News
- Streamlit GUI and Virtual Environment
- Pipfile of all requirements
- Dockfile for Docker deployment
- .yml file as a simple stack file to run and manage our app in a Docker Swarm
Our goal is to build a web app that implements all main features from our dataset. After, we deploy the app to a docker container that contains an image of our app. Lastly, docker swarm is implemented in order to prevent damage and have better speed.
By using the URL mode you can paste the URL you want and the text will be used automatically. By using the Paste mode you can copy-paste the article yourself.

- For prediction: we use pre-trained models stored in .pickle files and a Vectorizer also stored. We apply the Vectorizer and then wait for the prediction result from our models.
- For summarization: we use Summy Summarization
- For NLP: we make both a wordcloud which contains all the most common words in the article and a tablr with main features such as Lemma, Part of speech tags and Token (for each word).
- For NER checker: we use spacy's display to make all entities visual.


In order to run: streamlit run app.py
In order to deploy app: $ sudo docker build -t app:latest .
and: $ sudo docker run -p 8501:8501 app:latest
