import torch
import streamlit as st
import plotly.graph_objs as go
from google_trans_new import google_translator
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings

# available languages and supported embedding names
av_languages = {'English': {'word':'glove', 'elmo':'', 'flair':'mix', 'bert':'bert-base-uncased'},
                'Spanish': {'word':'es', 'flair':'es', 'bert':'bert-base-multilingual-uncased'},
                'Portuguese': {'word':'pt', 'elmo':'pt', 'flair':'pt', 'bert':'bert-base-multilingual-uncased'},
                'Italian': {'word':'it', 'flair':'it', 'bert':'bert-base-multilingual-uncased'},
                'French': {'word':'fr', 'flair':'fr', 'bert':'bert-base-multilingual-uncased'},
                'German': {'word':'de', 'flair':'de', 'bert':'bert-base-german-dbmdz-uncased'},
                'Japanese': {'word':'ja', 'flair':'ja', 'bert':'cl-tohoku/bert-base-japanese'},
                'Chinese': {'word':'zh', 'bert':'bert-base-chinese'},
                'Basque': {'word':'eu', 'elmo':["https://schweter.eu/cloud/eu-elmo/options.json", 
                           "https://schweter.eu/cloud/eu-elmo/weights.hdf5"], 'flair':'eu', 'bert':'bert-base-multilingual-uncased'}
               }

# example of main and similar sentences
example_sent = ["the doctor invited the patient for lunch",
                "the doctor didn't invite the patient for lunch",
                "the patient invited the doctor for lunch",
                "the surgeon invited the patient for lunch",
                "the doctor invited the patient for a meal",
                "the doctor and the patient went our for tea",
                "for patient the invited doctor lunch the",
                "a random sentence with two drops of sugar",
                "esta frase está en otro idioma"]


#@st.cache
def load_word_embeddings(ename):
    return DocumentPoolEmbeddings([WordEmbeddings(ename)])

#@st.cache
def load_flair_embeddings(ename):
    return DocumentPoolEmbeddings([FlairEmbeddings(f'{ename}-forward'), FlairEmbeddings(f'{ename}-backward')])

#@st.cache
def load_bert_embeddings(ename):
    # See BERT paper, section 5.3 and table 7 for layers
    return TransformerDocumentEmbeddings(ename, layers='-1,-2,-3,-4')

#@st.cache
def load_elmo_embeddings(ename):
    return DocumentPoolEmbeddings([ELMoEmbeddings(ename)])

def load_embeddings(lang, etype='word'):
    ename = av_languages[lang][etype]
    if etype == 'word':
        return load_word_embeddings(ename)
    if etype == 'flair':
        return load_flair_embeddings(ename)
    elif etype == 'elmo':
        return load_elmo_embeddings(ename)
    elif etype == 'bert':
        return load_bert_embeddings(ename)
    else:
        st.write('Error when loading embeddings Embedding type not recognized')


def calculate_similarities(gold, sim_sentences, embeddings):
    # compute current similarity values and score
    similarities = []
    q = Sentence(gold)
    embeddings.embed(q)
    
    for sent in sim_sentences:
        # if sentence is empty (user didn't fill it)
        if not sent:
            continue
    
        s = Sentence(sent)
        embeddings.embed(s)

        assert q.embedding.shape == s.embedding.shape
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        prox = cos(q.embedding, s.embedding)
        similarities.append(round(prox.item(), 4))
    return similarities

def sidebar():
    st.sidebar.markdown('''This app lets you see the difference in the contextual embeddings from different sentences. You can experiment with a `main` sentence and sentences similar to it, and check how similar contextual embeddings actually are.
    \nI added a number of languages, classic word embeddings and I use [Flair](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings) for the contextual embeddings: `flair`, `ELMo` and `BERT`. 
    If you would like to see another language or if there is a bug or error, feel free to open [an issue on the repo](https://github.com/anebz/eu-sim/issues).
    \n### Info on contextualized embeddings
    \nWord embeddings are mappings from a word to a vector of numbers, designed to capture its meaning. The idea is that synonyms have very similar embeddings, antonyms have opposite embedding and so on.
    \nContextual embeddings obtain the word mapping based on its context, capturing uses of words across varied contexts and encoding knowledge that transfers across languages. 
    \nTo obtain the embedding of the whole sentence, I use [DocumentEmbedding](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) from Flair to obtain a kind of average embedding of all the word embeddings in the sentence.
    \nFor more info, check out the [slides](https://github.com/anebz/eu-sim/blob/master/prev_data/semantic_search.pdf) of the talk I gave on the topic.
    \n### Findings
    \nFollowing my intuition, I expected that 'opposite' sentences would have 'opposite' embeddings. For example, `A invites B` is linguistically opposite to `B invites A`, or `A doesn't invite B`. I expected to see some difference in the embeddings.
    \nHowever, to my surprise all those sentences have very high embedding similarity. After doing the experiments, I concluded that this 'contrast' is very small compared to the fact that both sentences have many words in common, and this is what influences the embedding.
    \nThese are other things I found after experimenting with the embeddings:
    \n* As expected, word embeddings only capture individual words' meaning with no regard to context.
    \n* Sentences with the same subject and object have very similar embeddings, regardless of the verb or other parts of the sentence.
    \n* BERT embeddings don't seem to pay much attention to word order in the sentence, opposite to Flair and ELMo.
    \n* BERT embeddings are in general very very similar to the main sentence, even random sentences or sentences in other languages have a similarity of at least 70%. It is difficult to produce a sentence that produces a similarity of lower than 65%.
    \n\nFor more experiments and findings, check out the slides mentioned above. Feel free to chat with me on [Twitter](https://twitter.com/aberasategi) about your experiments or findings! 😄
    ''')
    return


def plot_data(similarities, checked_names):
    layout = go.Layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Similar sentence indexes: [0-7]')),
                       yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Embedding type')))
    st.plotly_chart(go.Figure(data=go.Heatmap(z=similarities, y=checked_names, colorscale='Blues'), layout=layout))
    return

if __name__ == "__main__":

    st.title('Contextual embedding similarity comparisor')
    sidebar()

    st.info("Streamlit has some problems with apps bigger than around 1GB, so multilanguage feature is disabled for now.")
    #lang = st.selectbox('Select language', tuple(av_languages.keys()))
    lang = st.selectbox('Select language', ['English'])

    # translate similar sentences to the chosen language
    if lang != 'English':
        translator = google_translator()
        example_sent = [translator.translate(sim, lang_tgt=av_languages[lang]['word']) for sim in example_sent]

    main_sent = st.text_input(f"Write your main sentence in {lang}", example_sent[0])
    
    st.write('Input similar sentences. Samples shown below')
    st.info('For languages other than English, the translations have been generated automatically and might contain errors')
    similar_sent = []
    for i in range(1, len(example_sent[1:]), 2):
        # display sample sentences in double column format for better readability
        cols = st.beta_columns(2)
        similar_sent.append(cols[0].text_input(f"Index {i-1}", example_sent[i], key=i))
        similar_sent.append(cols[1].text_input(f"Index {i}", example_sent[i+1], key=i+1))

    # display available embeddings in checkboxes
    st.subheader('Please choose at least one of the embeddings below')
    st.info('The first time it takes a while to load the embeddings')
    cols = st.beta_columns(len(av_languages[lang]))
    emb_boxes = [cols[i].checkbox(name) for i, name in enumerate(av_languages[lang])]

    if st.button('Run embedding comparison'):
        # check that the golden sentence isn't empty and that there is at least one similar sentence
        if main_sent and ''.join(similar_sent):
            similarities = []
            checked_names = []
            for name, box in zip(av_languages[lang], emb_boxes):
                if box:
                    checked_names.append(name)
                    # obtain similarity between gold and similar's embeddings
                    similarities.append(calculate_similarities(main_sent, similar_sent, load_embeddings(lang, name)))

            if similarities:
                plot_data(similarities, checked_names)
            else:
                st.write("Please select at least one embedding")
        else:
            st.write("Please write the golden sentence and at least one similar sentence")
