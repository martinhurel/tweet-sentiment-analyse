import os
import requests
import json
from datetime import date



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

response = requests.get('https://api.twitter.com/2/users/106062176/tweets', headers=headers, params=params)

data = json.loads(response.text)

tweets = []

for val in data['data']:
    tweets.append(val['text'])
