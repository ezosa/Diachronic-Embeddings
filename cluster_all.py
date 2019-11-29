import gensim
import re
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from collections import defaultdict
from scipy import spatial
import sys
import json
import time
import datetime

def timestamp (message):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print (st)
    print (message)
    return


def print_cluster(cluster, center, vocab, prev_vocab, words_of_interest): 
    outlist = []
    for c in sorted(cluster, key = lambda x: len(cluster[x]), reverse=True):
        
        cl = cluster[c]
        count = {}
        for w in cl:
            count[w] = vocab[w].count - prev_vocab[w].count if w in prev_vocab else vocab[w].count

        print (center[c], count[center[c]])
   
        
        hightlights = {w:count[w] for w in cl if w in words_of_interest}
        h_sort = sorted(hightlights, key = hightlights.get, reverse=True)
        res = [(center[c], count[center[c]])]
        res.extend([(h, count[h]) for h in h_sort if h != center[c]])        

        words = {w:count[w] for w in cl if not w in words_of_interest}
        w_sort = sorted(words, key = words.get, reverse=True)
        res.extend([(w,words[w]) for w in w_sort])
        
        for w in h_sort:
            if not w==center[c]: print (w, hightlights[w])
        for w in w_sort:
            if not w==center[c]: print (w, words[w])
        print ("\n")
        outlist.append(res)

    return outlist

def word_exist(word, model, prev_model):
    return (not prev_model
            or
            not word in prev_model.wv.vocab
            or
            model.wv.vocab[word].count > prev_model.wv.vocab[word].count)
        
def collect_words(model, prev_model):
    if lang == 'sv':
        regexp = "ism$|ismen$|ismens$"
    elif lang == 'svl':
        regexp = "ism$"
    elif lang == 'fi':
        regexp = "ismi$|ismin$|ismia$|ismissa$|ismista$|ismilla$|ismille$|ismilta$|ismina$|ismiksi$|ismitta$|ismit$|ismien$|ismeja$|ismeissa$|ismeista$|ismeihin$|ismeilla$|ismeilta$|ismille$|ismeina$|ismeiksi$|ismein$|ismeitta$|ismeineen$|ismiä$|ismissä$|ismistä$|ismillä$|ismiltä$|isminä$|ismittä$|ismejä$|ismeissä$|ismeistä$|ismeillä$|ismeiltä$|ismeinä$|ismeittä$"
    elif lang == "fil":
        regexp = "ismi$"

    min_len = 6 if lang in ["fi", "fil"] else 5
        
    return[w for w in model.wv.vocab if (len(w) >= min_len and re.search(regexp, w) and word_exist(w, model, prev_model))]            


def build_matrix(model, word_list, words_of_interest):
    if lang in ['fil', 'svl']:
        return np.array([[model.similarity(w1,w2) for w2 in word_list] for w1 in word_list])
        
    matrix = np.zeros((len(word_list), len(word_list)))
    for i in range(len(word_list)):
        for j in range(i, len(word_list)):
            w1 = word_list[i]
            w2 = word_list[j]
            if i == j:
                matrix[i,j] = 1.0
            # improves lemmatization:
            elif ((w1 in words_of_interest and w2 in words_of_interest)
                  and
                    ((w1.startswith(w2) and len(w2)>4)
                     or
                    (w2.startswith(w1) and len(w1)>4))):
                matrix[i,j] = matrix[j,i] = 0.99
            elif (lang in ["sv", "svl"] and
                  (w1[0:10] == 'rheumatism' and w2[0:9] == 'reumatism' or
                   w2[0:10] == 'rheumatism' and w1[0:9] == 'reumatism')):
                matrix[i,j] = matrix[j,i] = 0.99
            elif (lang in ["sv", "svl"] and
                  (w1[0:11] == 'katholicism' and w2[0:10] == 'katolicism' or
                   w2[0:11] == 'katholicism' and w1[0:10] == 'katolicism')):
                matrix[i,j] = matrix[j,i] = 0.99
            else:
                matrix[i,j] = matrix[j,i] = (1-spatial.distance.cosine(model[w1], model[w2]))
    return matrix


def cluster(model, prev_model, words_of_interest, thr=0.5):
    if thr == 1:
        word_list = words_of_interest
    else:
        word_list = [w for w in model.wv.vocab
                         if (word_exist(w, model, prev_model) and
                             any([model.wv.similarity(w, target)>thr for target in words_of_interest]))]


    timestamp("word_list: %d, words_of_interest: %d" %(len(word_list), len(words_of_interest)))
            
    matrix = build_matrix(model, word_list, words_of_interest)
    
    timestamp("matrix built, clustering")
    
    af = AffinityPropagation(affinity="precomputed").fit(matrix)
    
    return (word_list, af.labels_, af.cluster_centers_indices_)


def collect_clusters(clustering, words_of_interest):
    timestamp("clustering done, collecting clusters")
    word_list, labels, centers = clustering
    c2w = defaultdict(list)
    w2c = {}
    center = {}
    for i in range(len(labels)):
        if i in centers:
            center[labels[i]] = word_list[i]
        c2w[labels[i]].append(word_list[i])
        w2c[word_list[i]]=labels[i]

    clusters_of_interest = set([w2c[w] for w in words_of_interest])
    selected_clusters = {c:w for c,w in c2w.items() if c in clusters_of_interest}

    timestamp("selected words: %d" %sum([len(v) for v in selected_clusters.values()]))
    return selected_clusters, center



if __name__ == "__main__":
    try:
        lang = sys.argv[1]
        assert(lang in ['fi', 'fil', 'sv', 'svl'])
    except:
        print("usage: cluster_all.py [fi|fil|sv|svl] <output_json_path>")
        exit(1)


    if lang == 'fi':
        ys = ['1760', '1820', '1840', '1860', '1880', '1900']
        base_path =  "../models/FI_models/model_fi_"
    elif lang == 'fil':
        ys = ['1760', '1820', '1840', '1860', '1880', '1900']
        base_path =  "../models/FI_lemma/model_fi_"
    elif lang == 'sv':
        ys = ['1740', '1760', '1780', '1800', '1820', '1840', '1860', '1880', '1900']
        base_path =  "../models/SV_out_new/model_sv_"
    elif lang == 'svl':
        ys = ['1760', '1780', '1800', '1820', '1840', '1860', '1880', '1900']
        base_path =  "../models/SV_lowercase/model_sv_"

    
    prev_model = None

    res_dict = {}
    for y in ys:
        model_path = base_path +y+".w2v"
        model = gensim.models.Word2Vec.load(model_path)
        model.init_sims(replace=True)
        timestamp("\n*********************\n")
        print(y+"\n")
        words_of_interest = collect_words(model, prev_model)
        if not words_of_interest:
            print ("nothing")
            continue

        if lang in ['fi', 'fil'] and y == '1900':
            # too big model
            thr = 0.6
        else:
            thr = 0.5
            
        clustering = cluster(model, prev_model, words_of_interest, thr=thr)
    
        clusters, centers = collect_clusters(clustering, words_of_interest)

        print("\n")
        prev_vocab = prev_model.wv.vocab if prev_model else {}
        res_dict[y] = print_cluster(clusters, centers, model.wv.vocab, prev_vocab, words_of_interest)
    
        timestamp("done")

        prev_model = model
        
    try:
        json_path = sys.argv[2]
    except:
        json_path = 'cluster_enriched.json'

    with open(json_path, 'w') as jout:
        json.dump(res_dict, jout)
