import os
from transformers import pipeline
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

def make_spans(text,results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(split_in_sentences(text),results_list))
    return facts_spans

##Fiscal Sentiment by Sentence
fin_model= pipeline("sentiment-analysis", model='FinanceInc/auditor_sentiment_finetuned', tokenizer='FinanceInc/auditor_sentiment_finetuned')
def fin_ext(text):
    results = fin_model(split_in_sentences(text))
    return make_spans(text,results)
    
##Forward Looking Statement
def fls(text):
    fls_model = pipeline("text-classification", model="FinanceInc/finbert_fls", tokenizer="FinanceInc/finbert_fls")
    results = fls_model(split_in_sentences(text))
    return make_spans(text,results)


# finData = "US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month. The department expects inflation to continue to rise."
finData = """
We may be impacted by macroeconomic conditions resulting from the global COVID-19 pandemic.
Since the first quarter of 2020, there has been a worldwide impact from the COVID-19 pandemic. Government regulations and shifting social behaviors have, at times, limited or closed non-essential transportation, government functions, business activities and person-to-person interactions. Global trade conditions and consumer trends that originated during the pandemic continue to persist and may also have long-lasting adverse impact on us and our industries independently of the progress of the pandemic.

For example, pandemic-related issues have exacerbated port congestion and intermittent supplier shutdowns and delays, resulting in additional expenses to expedite delivery of critical parts. Similarly, increased demand for personal electronics has created a shortfall of semiconductors, which has caused challenges in our supply chain and production. In addition, labor shortages resulting from the pandemic, including worker absenteeism, has led to increased difficulty in hiring and retaining manufacturing and service workers, as well as increased labor costs and supplier delays. Sustaining our production trajectory will require the ongoing readiness and solvency of our suppliers and vendors, a stable and motivated production workforce and government cooperation, including for travel and visa allowances. The contingencies inherent in the ramp at new facilities such as Gigafactory Berlin-Brandenburg and Gigafactory Texas may be exacerbated by these challenges. Additionally, infection rates and regulations continue to fluctuate in various regions, which may impact operations. For example, in 2022, spikes in COVID-19 cases in Shanghai resulted in the temporary shutdown of Gigafactory Shanghai, as well as parts of our supply chain, and impacted our ability to deliver cars.
We cannot predict the duration or direction of current global trends or their sustained impact. Ultimately, we continue to monitor macroeconomic conditions to remain flexible and to optimize and evolve our business as appropriate, and attempt to accurately project demand and infrastructure requirements globally and deploy our production, workforce and other resources accordingly. Lastly, rising interest rates may lead to consumers to increasingly pull back spending, including on our products, which may harm our demand, business and operating results. If we experience unfavorable global market conditions, or if we cannot or do not maintain operations at a scope that is commensurate with such conditions or are later required to or choose to suspend such operations again, our business, prospects, financial condition and operating results may be harmed.
"""

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
    print(f"Negative sentences: {counterPosNeg['Negative']/numSentences*100:.02f}%")
    print(f"Neutral sentences: {counterPosNeg['Neutral']/numSentences*100:.02f}%")
    print(f"Positive sentences: {counterPosNeg['Positive']/numSentences*100:.02f}%\n")

    counterFls = {
        'Not FLS': 0,
        'Specific FLS': 0,
        'Non-specific FLS': 0
    }

    for sentence in flsClassified:
        counterFls[sentence[1]] += 1

    
    print(f"Not FLS sentences: {counterFls['Not FLS']/numSentences*100:.02f}%")
    print(f"Non-specific FLS sentences: {counterFls['Non-specific FLS']/numSentences*100:.02f}%")
    print(f"Specific FLS sentences: {counterFls['Specific FLS']/numSentences*100:.02f}%")
    

posNeg = fin_ext(finData)
print(posNeg, '\n')
flsClassified = fls(finData)
print(flsClassified, '\n')

printStats(posNeg, flsClassified)

