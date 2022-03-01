from typing import List
from matplotlib.pyplot import cohere
import pandas as pd
import numpy as np
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import cohen_kappa_score, adjusted_rand_score, silhouette_score

"""
calculate the best amount of topics for LSI model
"""
def best_num_topic(corpus, dictionary, max_n_topic)->List:
    np.random.seed(0)
    results = []

    for t in range(2, max_n_topic):
        lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=t)
        corpus_lsi = lsi_model[corpus]

        cm = CoherenceModel(model=lsi_model, corpus=corpus_lsi, coherence='u_mass')
        score = cm.get_coherence()
        tup = t, score
        results.append(tup)

    results = pd.DataFrame(results, columns=['topic', 'score'])
    # lowest score means the best
    s = pd.Series(results.score.values, index=results.topic.values)
    s.plot()

    return results

"""
calculate the best k for kmean based on silhouette score
"""
def best_silhouette_score(model, corpus, dictionary, max_n_clusters, best_num_topic = 2):
    np.random.seed(0)

    results = []
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=best_num_topic)
    corpus_lsi = lsi_model[corpus]

    for t in range(2, max_n_clusters):
 
        X = np.array([[tup[1] for tup in arr] for arr in corpus_lsi])
        if 'random_state' in model.get_params():
            model.set_params(random_state = 0)
        
        models = model.set_params(n_clusters = t).fit(X)

        score = silhouette_score(X, models.labels_)

        tup = t, score
        results.append(tup)
    
    results = pd.DataFrame(results, columns=['topic', 'score'])
    s = pd.Series(results.score.values, index=results.topic.values)
    _ = s.plot()

    return results

def get_kappa(actual, pred):
    kappa = cohen_kappa_score(actual, pred)
    print("kappa: {}".format(kappa))
    return kappa

def get_rand_score(actual, pred):
    rand_score= adjusted_rand_score(actual, pred)
    print("rand score: {}".format(rand_score))
    return rand_score

def get_silhouette_score(X, pred):
    score = silhouette_score(X, pred, random_state = 0)
    print("silhouette score: {}".format(score))
    return score

def get_coherence(cm):
    coherence = cm.get_coherence()
    print("coherence: {}".format(coherence))
    return coherence

def label_to_cluster_num(pred, k, books):
    pred_map = dict()
    offset = [(ord(label) - ord('a')) for label in books]

    for i in range(0, k):
        count = np.bincount(pred[i * 200 : i * 200 + 200])
        count.resize((1,k), refcheck=False) # resize row's dimension to 1xk
        count = count[0]
        index = np.where(count == max(count))[0][0]
        label =  chr(ord('a')+offset[i])
        pred_map[label] = index
        print(label + ":" + np.array2string(count) + "\t" + label + " mapped to " + str(index))

    return pred_map