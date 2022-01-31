from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np
import os
import requests
import json
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from flask import Response
from wordcloud import WordCloud
import matplotlib.pyplot as plt


app = Flask(__name__)

vocab_size = 10000
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

df = pd.read_csv('Tweets.csv')

#Suppression des doublons
df = df.drop_duplicates('tweet_id')

# Remove neutral rows
df = df.drop(df[df.airline_sentiment == 'neutral' ].index)

# Cleaning data set
def remove_useless_words_in_text(txt):
    mentions = re.findall("@([a-zA-Z0-9_]{1,50})", txt)
    for mention in mentions:
        txt = txt.replace(mention, '')
    txt = txt.replace('@', '')
    txt = txt.replace('https', '')
    txt = txt.replace('http', '')
    return txt

df['text'] = df['text'].transform(remove_useless_words_in_text)
    
sentences = [] # headlines
labels = [] # labels 
training_size = 1200
for idx,row in df.iterrows():
    if row['airline_sentiment_confidence'] > 0.7:
        sentences.append(row['text'])
        if row['airline_sentiment'] == 'positive':
            labels.append(1)
        elif row['airline_sentiment'] == 'negative':
            labels.append(0)

training_sentences = sentences[0:training_size]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)


class Analyse:
    def __init__(self, data):
        self.data = data


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/plot-sentiment-repartition.png')
def plot_png(): 

    plt.style.use('seaborn')
    df = pd.read_csv('df_tweets.csv')
    positif = len(df[df['Sentiment'] > 0.5])
    negatif = len(df[df['Sentiment'] < 0.5])
    fig = Figure()

    text = ''
    for idx,row in df.iterrows():
        text = text + ' ' +  remove_useless_words_in_text(row['Tweet'])

    # Create the wordcloud object
    wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

    fig = Figure(figsize=(14,4))
    axis1 = fig.add_subplot(1, 3, 1)
    axis2 = fig.add_subplot(1, 3, 2)
    axis3 = fig.add_subplot(1, 3, 3)

    axis1.set_title('Tweet sentiment repartition')
    axis1.bar(['positif', 'negatif'], [positif, negatif], color = ['green', 'red'])
    
    axis2.imshow(wordcloud, interpolation='bilinear')
    axis2.axis('off')
    axis2.set_title('Word Cloud')
    
    axis3.boxplot(df['Sentiment'])
    axis3.set_title('Sentiment distribution')
    axis3.set_xticklabels(['Sentiment'])

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/', methods=['POST'])
def post_search():

    company_id = request.form.get('phrase')

    df_tweets = get_dataframe(company_id)

    df_tweets = df_tweets[df_tweets['Langue'] == 'en']

    mean_note = df_tweets['Sentiment'].mean()

    mean_note = "{:.0%}".format(mean_note)

    text_tweet = df_tweets['Tweet']

    df_tweets.to_csv('df_tweets.csv')

    return render_template('view.html', prediction=Analyse([mean_note, company_id, len(df_tweets),text_tweet]))

def get_sentiment_analyse(sentence):
    model = keras.models.load_model('/Users/martinhurel/Desktop/tweet-sentiment-analyse/price_prediction_model.h5')

    new_sequences = tokenizer.texts_to_sequences([sentence])
    # padding the new sequences to make them have same dimensions
    new_padded = pad_sequences(new_sequences, maxlen = max_length,padding = padding_type,truncating = trunc_type)

    new_padded = np.array(new_padded)

    val_analyse = model.predict(new_padded)

    return val_analyse

def get_dataframe(company_id):

    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAOzHYgEAAAAAWRcWlM9yvQqIO%2Fcp3kshZvYgT28%3Dc7xWTq9cAukE6gRlRJJuu1pgxpQz7gJqwgWkQywz5M7Twz3R68'

    headers = {
        'Authorization': f"Bearer {BEARER_TOKEN}",
    }

    params = (
        ('start_time', '2022-01-01T00:00:00.000Z'),
        ('end_time', '2022-01-31T00:00:00.000Z'),
        ('tweet.fields', 'text,lang'),
    )

    response = requests.get('https://api.twitter.com/2/users/'+ company_id +'/mentions', headers=headers, params=params)

    data = json.loads(response.text)

    df = pd.DataFrame(columns = ['Tweet', 'Sentiment'])

    for val in data['data']:
        df = df.append({
            'Langue': val['lang'],
            'Tweet': val['text'],
            'Sentiment': get_sentiment_analyse(val['text'])[0][0],
        }, ignore_index=True)

    return df



if __name__ == '__main__':
    app.run(debug=True)
