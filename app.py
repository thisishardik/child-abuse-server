from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from flask import Flask, request, jsonify
import pickle
import nltk

app = Flask(__name__)

nltk.download('punkt')
nltk.download('vader_lexicon')


@app.route("/predict", methods=["GET"])
def predict():
    args = request.args
    sentence = args.get("sentence")

    sia = SentimentIntensityAnalyzer()
    sa = SentimentAnalyzer()

    preds = sia.polarity_scores(sentence)

    compound = preds['compound']

    if(compound == 0):
        sev = "Moderate"
    else:
        compound = compound * 10
        if(compound >= 3):
            sev = "Low"
        elif compound >= 1 and compound < 3:
            sev = "Moderate"
        elif compound >= -2 and compound < 0:
            sev = "High"
        else:
            sev = "Critical"

    # print(compound)
    # print(sev)
    return {"severity": f"{sev}"}


if __name__ == "__main__":
    app.run()
