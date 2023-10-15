# , AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import os
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import streamlit as st
# , AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline

# os.system("pip install gradio==3.0.18")
# import gradio as gr
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

st.title('Financial Document Info')

# create dropdown
documents = ['Nvidia Stocks', 'Tesla 10k', 'Google Financial Report', 'Custom']
document_select = st.selectbox(
    'Select a financial document to analyze', documents)

# if (document_select == 'document 1'):
# overwrite the finData text area with the text from document 1

# set finData to the user input
# finData = st.text_area('Enter text here', height=200)

# default content for finData
finData = ""

# check the selected document and update finData accordingly
if document_select == 'Tesla 10k':
    with open('tesla.txt', 'r') as f:
        finData = f.read()
elif document_select == 'Nvidia Stocks':
    with open("nvidia.txt", "r") as f:
        finData = f.read()
elif document_select == 'Google Financial Report':
    with open("google.txt", "r") as f:
        finData = f.read()
elif document_select == 'Custom':
    finData = ""


# display selected document content in the text area
finData = st.text_area('Enter text here', value=finData, height=200)

# create a streamlit butotn that calls the printStats function
output = st.button(
    'Run Sentiment Analysis and Forward Looking Statement Analysis')


def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent) for sent in doc.sents]


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


posNeg = fin_ext(finData)
flsClassified = fls(finData)


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

    # st.write(f"Negative sentences: {percentNeg:.02f}%")
    # st.write(f"Neutral sentences: {percentNeu:.02f}%")
    # st.write(f"Positive sentences: {percentPos:.02f}%\n")
    label_styles = {
        'Positive': 'background-color: #99ff99;',
        'Specific FLS': "background-color: #99ff99;",
        'Neutral': 'background-color: #dddddd;',
        'Non-specific FLS': 'background-color: #dddddd;',
        'Negative': 'background-color:  #ff8080;',
        'Not FLS': 'background-color:  #ff8080;'
    }

    # Positive/Neutral/Negative Sentences

    st.header("Positive/Neutral/Negative Sentences:")
    postneg_output = ""
    for text, label in posNeg:
        # Apply CSS styles based on the label
        label_style = f"{label_styles.get(label, '')}"
        postneg_output += f'<span style="{label_style}">{text} </span>'
    st.expander("Click to see Positive/Neutral/Negative Sentences").markdown(
        postneg_output, unsafe_allow_html=True)

    print("POSTNEG", posNeg, '\n')

    # Labels and values for the pie chart
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = [percentNeg, percentNeu, percentPos]
    colors = ['#ff8080', '#dddddd', '#99ff99']
    explode = (0.1, 0, 0)  # explode the 1st slice (i.e., 'Negative')

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.show()
    st.pyplot(plt)

    # Forward Leaning Statements

    st.header("Forward Leaning Statements:")
    fin_output = ""
    for text, label in flsClassified:
        # Apply CSS styles based on the label
        label_style = label_styles.get(label, '')
        fin_output += f'<span style="{label_style}">{text} </span>'

    st.expander("Click to see Forward Leaning Statements").markdown(
        fin_output, unsafe_allow_html=True)

    counterFls = {
        'Not FLS': 0,
        'Specific FLS': 0,
        'Non-specific FLS': 0
    }

    for sentence in flsClassified:
        counterFls[sentence[1]] += 1

    # plot 'not fls', 'specific fls', and 'non-specific fls' as a bar chart using matplotlib
    categories = list(counterFls.keys())
    values = list(counterFls.values())

    # bar_colors = {
    #     'Non-specific FLS':  (221, 221, 221, 1),
    #     'Specific FLS': (153, 255, 153, 1),
    #     'Not FLS': (255, 128, 128, 1),
    # }

    plt.figure(figsize=(8, 6))
    # plt.bar(categories, values, color=[bar_colors[category] for category in categories])
    plt.bar(categories, values, color=['#ff8080', '#99ff99', '#dddddd'])
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart for FLS Categories')
    plt.show()
    st.pyplot(plt)

    # print(f"Not FLS sentences: {counterFls['Not FLS']/numSentences*100:.02f}%")
    # print(
    #     f"Non-specific FLS sentences: {counterFls['Non-specific FLS']/numSentences*100:.02f}%")
    # print(
    #     f"Specific FLS sentences: {counterFls['Specific FLS']/numSentences*100:.02f}%")

    st.write(counterFls)


# results = f'<table><tr><th>Positive/Neutral/Negative Sentences</th><th>Forward Leaning Statements:</th><tr><td>{postneg_output}</td><td>{fin_output}</td></tr></table>'

# st.markdown(results, unsafe_allow_html=True)
if output:
    if finData != "":
        printStats(posNeg, flsClassified)
