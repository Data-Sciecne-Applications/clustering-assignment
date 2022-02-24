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
    from sklearn.metrics import silhouette_score
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

def get_silhouette_score(vectorizer, pred):
    X = np.array([[tup[1] for tup in arr] for arr in vectorizer])
    score = silhouette_score(X, pred, random_state = 0)
    print("silhouette score: {}".format(score))
    return score

def get_coherence(cm):
    coherence = cm.get_coherence()
    print("coherence: {}".format(coherence))
    return coherence