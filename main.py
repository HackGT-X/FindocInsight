# , AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import os
from transformers import pipeline
import spacy
import streamlit as st
import pandas as pd
import numpy as np

# , AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import en_core_web_sm
import os
import matplotlib.pyplot as plt

# os.system("pip install gradio==3.0.18")
# import gradio as gr
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

st.title('Financial Document Info')


def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]


def make_spans(text, results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(split_in_sentences(text), results_list))
    return facts_spans


# Fiscal Sentiment by Sentence
fin_model = pipeline("sentiment-analysis", model='FinanceInc/auditor_sentiment_finetuned',
                     tokenizer='FinanceInc/auditor_sentiment_finetuned')


def fin_ext(text):
    results = fin_model(split_in_sentences(text))
    return make_spans(text, results)

# Forward Looking Statement


def fls(text):
    fls_model = pipeline(
        "text-classification", model="FinanceInc/finbert_fls", tokenizer="FinanceInc/finbert_fls")
    results = fls_model(split_in_sentences(text))
    return make_spans(text, results)


# set finData to the user input
finData = st.text_area('Enter text here', height=200)

posNeg = fin_ext(finData)
findFls = fls(finData)


def printStats(posNeg, flsClassified):
    counterPosNeg = {
        'Negative': 0,
        'Neutral': 0,
        'Positive': 0
    }
    for sentence in posNeg:
        counterPosNeg[sentence[1]] += 1

    numSentences = len(posNeg)

    percentNeg = counterPosNeg['Negative'] / numSentences * 100
    percentNeu = counterPosNeg['Neutral'] / numSentences * 100
    percentPos = counterPosNeg['Positive'] / numSentences * 100

    st.write(f"Negative sentences: {percentNeg:.02f}%")
    st.write(f"Neutral sentences: {percentNeu:.02f}%")
    st.write(f"Positive sentences: {percentPos:.02f}%\n")

    # Labels and values for the pie chart
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = [percentNeg, percentNeu, percentPos]
    colors = ['red', 'gray', 'green']
    explode = (0.1, 0, 0)  # explode the 1st slice (i.e., 'Negative')

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.show()

    st.pyplot(plt)

    counterFls = {
        'Not FLS': 0,
        'Specific FLS': 0,
        'Non-specific FLS': 0
    }

    for sentence in flsClassified:
        counterFls[sentence[1]] += 1

    # print(f"Not FLS sentences: {counterFls['Not FLS']/numSentences*100:.02f}%")
    # print(
    #     f"Non-specific FLS sentences: {counterFls['Non-specific FLS']/numSentences*100:.02f}%")
    # print(
    #     f"Specific FLS sentences: {counterFls['Specific FLS']/numSentences*100:.02f}%")

    st.write(counterFls)


posNeg = fin_ext(finData)
print(posNeg, '\n')
flsClassified = fls(finData)
print(flsClassified, '\n')

# create a streamlit butotn that calls the printStats function
output = st.button(
    'Run Sentiment Analysis and Forward Looking Statement Analysis')
if output:
    printStats(posNeg, flsClassified)
