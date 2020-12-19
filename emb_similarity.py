from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot

import torch
import pandas as pd
import streamlit as st


@st.cache
def load_flair_embeddings():
    flair_emb = [(DocumentPoolEmbeddings([FlairEmbeddings("eu-forward"), FlairEmbeddings("eu-backward")]),
                DocumentPoolEmbeddings([FlairEmbeddings('mix-forward'), FlairEmbeddings('mix-backward')]))]
    return flair_emb

@st.cache
def load_bert_cased_embeddings():
    # See BERT paper, section 5.3 and table 7
    bert_layers = '-1,-2,-3,-4'
    bert_cased_emb = [TransformerDocumentEmbeddings('bert-base-multilingual-cased', layers=bert_layers),
                TransformerDocumentEmbeddings('bert-base-cased', layers=bert_layers)]
    return bert_cased_emb

@st.cache
def load_bert_uncased_embeddings():
    # See BERT paper, section 5.3 and table 7
    bert_layers = '-1,-2,-3,-4'
    bert_uncased_emb = [TransformerDocumentEmbeddings('bert-base-multilingual-uncased', layers=bert_layers),
                TransformerDocumentEmbeddings('bert-base-uncased', layers=bert_layers)]
    return bert_uncased_emb

@st.cache
def load_elmo_embeddings():
    elmo_emb = [(DocumentPoolEmbeddings([ELMoEmbeddings(options_file="https://schweter.eu/cloud/eu-elmo/options.json", 
                                                                    weight_file="https://schweter.eu/cloud/eu-elmo/weights.hdf5")]),
                DocumentPoolEmbeddings([ELMoEmbeddings()]))]
    return elmo_emb


def calculate_similarities(gold, sim_sentences, embeddings):
    # compute current similarity values and score
    similarities = []
    q = Sentence(gold)
    embeddings.embed(q)
    
    for i in range(len(sim_sentences)):
        
        s = Sentence(sim_sentences[i])
        embeddings.embed(s)

        assert q.embedding.shape == s.embedding.shape
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        prox = cos(q.embedding, s.embedding)
        similarities.append(round(prox.item(), 4))
        
    return similarities


if __name__ == "__main__":

    st.title('Sentence embedding similarity comparisor')
    option = st.selectbox('Choose language pair', ('Basque - English', 'Home phone', 'Mobile phone'))
    lang1, lang2 = option.split('-')

    # capture gold and similar sentences
    example_sentences = ["The doctor invited the patient for lunch",
                        "the surgeon invited the patient for lunch", 
                        "the doctor invited the doctor for lunch",
                        "the professor invited the patient for lunch",
                        "the doctor invited the patient for a meal",
                        "the doctor took the patient out for tea",
                        "the doctor paid for the patient's lunch",
                        "the park was empty at this time of night",
                        "das ergibt doch keinen Sinn"]
    gold_sent_en = st.text_input(f"Gold sentence in {lang2}", example_sentences[0])
    st.subheader('Input similar sentences')
    similar_sent_en = [st.text_input('', example) for example in example_sentences[1:]]


    embedding_names = ["flair", "bert_cased", "bert_uncased", "elmo"]

    # obtain similarity between gold and similar's embeddings
    st.subheader('Please choose at least one of the embeddings below')
    cols = st.beta_columns(4)
    flair = cols[0].checkbox("Flair")
    bertc = cols[1].checkbox("BERT cased")
    bertu = cols[2].checkbox("BERT uncased")
    elmo = cols[3].checkbox("ELMo")

    if st.button('Run embedding comparison'):

        if flair:
            embeddings_all = load_flair_embeddings()

        try:
            data = [calculate_similarities(gold_sent_en, similar_sent_en, emb) for emb in embeddings_all[0]]
            st.bar_chart(pd.DataFrame(data[0], columns=['flair']))
        except:
            st.write("Please select at least one embedding")
