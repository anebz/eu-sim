from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot

import torch
import streamlit as st


@st.cache
def load_flair_embeddings(embeddings_all):
    # load flair embeddings
    embeddings_all[0].append(DocumentPoolEmbeddings([FlairEmbeddings("eu-forward"), FlairEmbeddings("eu-backward")]))
    embeddings_all[1].append(DocumentPoolEmbeddings([FlairEmbeddings('mix-forward'), FlairEmbeddings('mix-backward')]))
    return embeddings_all


@st.cache
def load_bert_embeddings(embeddings_all):
    # load BERT embeddings
    # See BERT paper, section 5.3 and table 7
    #bert_layers = list(range(-9, -13, -1))
    bert_layers = '-1,-2,-3,-4'
    bert_type = 'base' # 'large'

    # BERT cased
    embeddings_all[0].append(TransformerDocumentEmbeddings('bert-base-multilingual-cased', layers=bert_layers))
    embeddings_all[1].append(TransformerDocumentEmbeddings('bert-'+bert_type+'-cased', layers=bert_layers))

    # BERT uncased
    embeddings_all[0].append(TransformerDocumentEmbeddings('bert-base-multilingual-uncased', layers=bert_layers))
    embeddings_all[1].append(TransformerDocumentEmbeddings('bert-'+bert_type+'-uncased', layers=bert_layers))
    return embeddings_all


@st.cache
def load_elmo_embeddings(embeddings_all):
    # load ELMo embeddings
    embeddings_all[0].append(DocumentPoolEmbeddings([ELMoEmbeddings(options_file="https://schweter.eu/cloud/eu-elmo/options.json", 
                                                                    weight_file="https://schweter.eu/cloud/eu-elmo/weights.hdf5")]))
    embeddings_all[1].append(DocumentPoolEmbeddings([ELMoEmbeddings()]))
    return embeddings_all


def calculate_similarities(gold, sim_sentences, embeddings):
    similarities = []
    q = Sentence(gold)
    embeddings.embed(q)
    score = 0
    
    for i in range(len(sim_sentences)):
        
        s = Sentence(sim_sentences[i])
        embeddings.embed(s)

        assert q.embedding.shape == s.embedding.shape
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        prox = cos(q.embedding, s.embedding)
        similarities.append(round(prox.item(), 4))
        
        if i > 0 and similarities[i] <= similarities[i-1]:
            score += 1

    # return current similarity values and score to the global data structures
    return (similarities, int(score/float(len(sim_sentences))*100))


def plot_similarities(gold_sent_en, similar_sent_en, data, embedding_names):
    st.write(gold_sent_en + '\n')
    for i, sent in enumerate(similar_sent_en):
        # print each similar sentence
        st.write(f"{i}. {sent}")
        for emb_name in embedding_names:
            # print each similarity value for each variant
            st.write(f"\t {emb_name} similarity: {data[i][0]}")
    # print scores for all variants
    st.write("Scores: ", ", ".join(f'{embed}: {scor}%' for embed, scor in zip(embedding_names, data[i][1])))
    
    '''
    # plot similarity heatmap
    trace = go.Heatmap(z=similarities_all[i], y=embedding_names, colorscale='Blues')
    fig = plotly.tools.make_subplots(rows=1, cols=10)
    fig.append_trace(trace, 1, i+1)
    iplot([trace], filename='basic-heatmap' + str(i))
    '''
    return

if __name__ == "__main__":

    st.title('Sentence embedding similarity comparisor')
    st.subheader('Embeddings')

    # [[flair_eu, bert_cased_eu, bert_uncased eu, elmo_eu], [flair_en, bert_cased_en, bert_uncased en, elmo_en]]
    embeddings_all = [[], []]
    embedding_names = ["flair", "bert_cased", "bert_uncased", "elmo"]
    #embedding_names = ["bert4c", "bert4u", "bert3c", "bert3u", "bert2c","bert2u", "bert1c", "bert1u"]

    option = st.selectbox('Choose language pair', ('Basque - English', 'Home phone', 'Mobile phone'))
    lang1, lang2 = option.split('-')

    st.subheader('Please choose at least one of the embeddings below')
    cols = st.beta_columns(3)
    bflair = cols[0].button('Flair embeddings')
    bbert = cols[1].button('BERT embeddings')
    belmo = cols[2].button('ELMo embeddings')

    if bflair:
        embeddings_all = load_flair_embeddings(embeddings_all)
    if bbert:
        embeddings_all = load_bert_embeddings(embeddings_all)
    if belmo:
        embeddings_all = load_elmo_embeddings(embeddings_all)

    # capture gold and similar sentences
    example_sentences = ["The doctor invited the patient for lunch",
                        "the surgeon invited the patient for lunch", 
                        "the doctor invited the doctor for lunch",
                        "the professor invited the patient for lunch",
                        "the doctor invited the patient for a meal",
                        "the doctor took the patient out for tea",
                        "the doctor paid for the patient's lunch"]
    gold_sent_en = st.text_input(f"Gold sentence in {lang2}", example_sentences[0])
    st.subheader('Input similar sentences')
    similar_sent_en = [st.text_input('', example) for example in example_sentences[1:]]

    # obtain similarity between gold and similar's embeddings
    if st.button('Run embedding comparison'):
        data = []
        for i in range(len(embeddings_all[0])):
            st.write(f"Calculating {embedding_names[i]} embeddings for english")
            data.append(calculate_similarities(gold_sent_en, similar_sent_en, embeddings_all[1][i]))
        
        plot_similarities(gold_sent_en, similar_sent_en, data, embedding_names)
