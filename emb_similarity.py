import torch

from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings, FlairEmbeddings, ELMoEmbeddings, TransformerWordEmbeddings

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot


def load_flair_embeddings(embeddings_all):
    # load flair embeddings
    embeddings_all[0].append(DocumentPoolEmbeddings([FlairEmbeddings("eu-forward"), FlairEmbeddings("eu-backward")]))
    embeddings_all[1].append(DocumentPoolEmbeddings([FlairEmbeddings('mix-forward'), FlairEmbeddings('mix-backward')]))
    return embeddings_all


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

    # bert_layer case
    '''
    for layer in bert_layers:

        # BERT cased
        embeddings_all[0].append(DocumentPoolEmbeddings([TransformerWordEmbeddings('bert-base-multilingual-cased', layers=str(layer))]))
        embeddings_all[1].append(DocumentPoolEmbeddings([TransformerWordEmbeddings('bert-'+bert_type+'-cased', layers=str(layer))]))

        # BERT uncased
        embeddings_all[0].append(DocumentPoolEmbeddings([TransformerWordEmbeddings('bert-base-multilingual-uncased', layers=str(layer))]))
        embeddings_all[1].append(DocumentPoolEmbeddings([TransformerWordEmbeddings('bert-'+bert_type+'-uncased', layers=str(layer))]))
    '''
    return embeddings_all


def load_elmo_embeddings(embeddings_all):
    # load ELMo embeddings
    embeddings_all[0].append(DocumentPoolEmbeddings([ELMoEmbeddings(options_file="https://schweter.eu/cloud/eu-elmo/options.json", 
                                                                    weight_file="https://schweter.eu/cloud/eu-elmo/weights.hdf5")]))
    embeddings_all[1].append(DocumentPoolEmbeddings([ELMoEmbeddings()]))
    return embeddings_all


def get_gold_sentences(filename):
    gold_sentences = []
    with open(filename, 'rt') as f_p:
        for line in f_p:
            if line.startswith('"origin"'): # header
                continue
            
            if not line:
                continue
            
            line = line.rstrip().replace('"', '').split('\t')
            gold = line[0]
            sim_sentences = line[1:11]
            
            if gold:
                gold_sentences.append({gold: sim_sentences})
    return gold_sentences


def initialize_vectors(sent):
    similarities_all = []
    for i in range(len(sent)):
        similarities_all.append([])

    scores_all = []
    for i in range(len(sent)):
        scores_all.append([])
    return similarities_all, scores_all


def calculate_similarities(gold, sim_sentences, embeddings):
    similarities = []
    query = gold

    q = Sentence(query)
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
    return similarities, int(score/float(len(sim_sentences))*100)


def calculate(gold_sentences, embeddings, similarities_all, scores_all):
    for i in range(len(gold_sentences)):
        
        # obtain gold sentence and similar sentences from global list
        gold = list(gold_sentences[i].keys())[0]
        sim_sentences = gold_sentences[i][gold]
        
        # Calculate similarities for each 'gold' sentence and accumulated score
        similarities, score = calculate_similarities(gold, sim_sentences, embeddings)

        # append current similarity values and score to the global data structures
        scores_all[i].append(score)
        similarities_all[i].append(similarities)
    return similarities_all, scores_all


def plot_similarities(sent, similarities_all, scores_all):
    origin = list(sent[i].keys())[0]
    # print origin sentence
    print(origin + '\n')
    for j in range(len(sent[i][origin])):
        # print each similar sentence
        print(f"{j}. {sent[i][origin][j]}")
        for k in range(len(embedding_names)):
            # print each similarity value for each variant
            print(f"\t {embedding_names[k]} similarity: {similarities_all[i][k][j]}")
    # print scores for all variants
    print(f"Scores: " + ", ".join(f"{embed}: {scor}%" for embed, scor in zip(embedding_names, scores_all[i])))
    
    # plot similarity heatmap
    trace = go.Heatmap(z=similarities_all[i], y=embedding_names, colorscale='Blues')
    data=[trace]
    fig.append_trace(trace, 1, i+1)

    iplot(data, filename='basic-heatmap' + str(i))
    return

if __name__ == "__main__":
    
    # [[flair_eu, bert_cased_eu, bert_uncased eu, elmo_eu], [flair_en, bert_cased_en, bert_uncased en, elmo_en]]
    embeddings_all = [[], []]
    embedding_names = ["flair", "bert_cased", "bert_uncased", "elmo"]
    #embedding_names = ["bert4c", "bert4u", "bert3c", "bert3u", "bert2c","bert2u", "bert1c", "bert1u"]

    embeddings_all = load_flair_embeddings(embeddings_all)
    embeddings_all = load_bert_embeddings(embeddings_all)
    embeddings_all = load_elmo_embeddings(embeddings_all)

    test_eu = "goldstandard_eu_lexicover.tsv"
    test_en = "goldstandard_en_lexicover.tsv"

    sent_eu = get_gold_sentences(test_eu)
    sent_en = get_gold_sentences(test_en)

    similarities_all_eu, scores_all_eu = initialize_vectors(sent_eu)
    similarities_all_en, scores_all_en = initialize_vectors(sent_en)


    for i in range(len(embeddings_all[0])):
        print(f"Calculating {embedding_names[i]} embeddings for basque")
        similarities_all_eu, scores_all_eu = calculate(sent_eu, embeddings_all[0][i], similarities_all_eu, scores_all_eu)
        print(f"Calculating {embedding_names[i]} embeddings for english")
        similarities_all_en, scores_all_en = calculate(sent_en, embeddings_all[1][i], similarities_all_en, scores_all_en)

    fig = plotly.tools.make_subplots(rows=1, cols=10)

    for i in range(len(sent_en)):
        print(f"\nSentence #{i}")
        print("Basque")
        plot_similarities(sent_eu, similarities_all_eu, scores_all_eu)
        print("English")
        plot_similarities(sent_en, similarities_all_en, scores_all_en)
