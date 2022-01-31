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

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.set_xlabel('time (s)')
    axis.set_title('subplot 2')
    axis.set_ylabel('Undamped')
    axis.plot(xs, ys)
    return fig

@app.route('/', methods=['POST'])
def post_search():

    company_id = request.form.get('phrase')

    df_tweets = get_dataframe(company_id)

    mean_note = df_tweets['Sentiment'].mean()

    mean_note = "{:.0%}".format(mean_note)

    return render_template('view.html', prediction=Analyse([mean_note, company_id, len(df_tweets)]))

def get_sentiment_analyse(sentence):
    model = keras.models.load_model('/Users/martinhurel/Desktop/tweet-sentiment-analyse/price_prediction_model.h5')

    new_sequences = tokenizer.texts_to_sequences([sentence])
    # padding the new sequences to make them have same dimensions
    new_padded = pad_sequences(new_sequences, maxlen = max_length,

                            padding = padding_type,
                            truncating = trunc_type)

    new_padded = np.array(new_padded)

    val_analyse = model.predict(new_padded)

    return val_analyse

def get_dataframe(company_id):

    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAOzHYgEAAAAAWRcWlM9yvQqIO%2Fcp3kshZvYgT28%3Dc7xWTq9cAukE6gRlRJJuu1pgxpQz7gJqwgWkQywz5M7Twz3R68'

    headers = {
        'Authorization': f"Bearer {BEARER_TOKEN}",
    }

    params = (
        ('exclude', 'replies,retweets'),
        ('start_time', '2022-01-01T00:00:00.000Z'),
        ('end_time', '2022-01-31T00:00:00.000Z'),
        ('tweet.fields', 'text'),
    )

    response = requests.get('https://api.twitter.com/2/users/'+ company_id +'/tweets', headers=headers, params=params)

    data = json.loads(response.text)

    df = pd.DataFrame(columns = ['Tweet', 'Sentiment'])

    for val in data['data']:
        df = df.append({
            'Tweet': val['text'],
            'Sentiment': get_sentiment_analyse(val['text'])[0][0],
        }, ignore_index=True)

    return df



if __name__ == '__main__':
    app.run(debug=True)
