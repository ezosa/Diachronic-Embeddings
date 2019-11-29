import gensim
import re
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from collections import defaultdict
from scipy import spatial
import sys
import json

def build_matrix(model, word_list, counts):
    matrix = np.zeros((len(word_list), len(word_list)))
    for i in range(len(word_list)):
        for j in range(i, len(word_list)):
            if i == j:
                matrix[i,j] = 1.0
            elif word_list[i].startswith(word_list[j]) or word_list[j].startswith(word_list[i]):
                matrix[i,j] = matrix[j,i] = 0.99
            elif (lang in ['sv', 'svl', 'svk'] and
                  (word_list[i][0:10] == 'rheumatism' and word_list[j][0:9] == 'reumatism' or
                  word_list[j][0:10] == 'rheumatism' and word_list[i][0:9] == 'reumatism')):
                matrix[i,j] = matrix[j,i] = 0.99
            elif (lang in ['sv', 'svl', 'svk'] and
                  (word_list[i][0:11] == 'katholicism' and word_list[j][0:10] == 'katolicism' or
                  word_list[j][0:11] == 'katholicism' and word_list[i][0:10] == 'katolicism')):
                matrix[i,j] = matrix[j,i] = 0.99
            else:
                matrix[i,j] = matrix[j,i] = (1-spatial.distance.cosine(model[word_list[i]], model[word_list[j]]))
    return matrix


def cluster(model, prev_model):
    model.init_sims(replace=True)

    ism = {}
    for k,v in model.wv.vocab.items():
        if lang in ['sv', 'svl', 'svk']:
            regexp = "ism$|ismen$|ismens$"
        else:
            regexp = "ismi$|ismin$|ismia$|ismissa$|ismista$|ismilla$|ismille$|ismilta$|ismina$|ismiksi$|ismitta$|ismit$|ismien$|ismeja$|ismeissa$|ismeista$|ismeihin$|ismeilla$|ismeilta$|ismille$|ismeina$|ismeiksi$|ismein$|ismeitta$|ismeineen$|ismiä$|ismissä$|ismistä$|ismillä$|ismiltä$|isminä$|ismittä$|ismejä$|ismeissä$|ismeistä$|ismeillä$|ismeiltä$|ismeinä$|ismeittä$"

        min_len = 6 if lang in ["fi", "fil"] else 5
            
        if len(k) >= min_len and re.search(regexp, k):
            count = v.count
            if k in prev_model.wv.vocab:
                count = count - prev_model.wv.vocab[k].count
            if count > 0:
                ism[k] = count
            
    word_list = list(ism)
    if len(word_list) == 0:
        return None
    
    matrix = build_matrix(model, word_list, ism)

    af = AffinityPropagation(affinity="precomputed", verbose=False).fit(matrix)
    
    return (word_list, ism, af.labels_, af.cluster_centers_indices_)

def print_cluster(word_list, counts, labels, centers):
    cluster = defaultdict(list)
    center = {}
    for i in range(len(labels)):
        if i in centers:
            center[labels[i]] = word_list[i]     
        cluster[labels[i]].append(word_list[i])

    for c in sorted(cluster, key = lambda x: len(cluster[x]), reverse=True):
        cl = cluster[c]
        print (center[c], counts[center[c]])
        words = {w:count for w,count in counts.items() if w in cl}
        for w in sorted(words, key=words.get, reverse=True):
            if not w==center[c]:
                print (w, words[w])
        print ("\n")


def make_json(word_list, counts, labels, centers):
    # 1 most frequent
    outlist = []
    # 2 most frequent
    outlist2 = []
    # most freq + centroid
    outlist_cf = []
    
    cluster = defaultdict(list)
    center = {}
    for i in range(len(labels)):
        if i in centers:
            center[labels[i]] = word_list[i]     
        cluster[labels[i]].append(word_list[i])

    for c in sorted(cluster, key = lambda x: len(cluster[x]), reverse=True):
        cl = cluster[c]
        words = {w:count for w,count in counts.items() if w in cl}
        res = [w for w in sorted(words, key=words.get, reverse=True)]
        if len(res) > 2:
            res2 = [res[0]+"_"+res[1]] + res[2:]
            res_noc = [r for r in res if r != center[c]]
            res_cf = [center[c] + "_" + res_noc[0]] + res_noc[2:]
        elif len(res) == 2:
            res2 = res_cf = [res[0] + "_" + res[1]]
        else:
            res2 = res_cf = res
            
            
        outlist.append(res)
        outlist2.append(res2)
        outlist_cf.append(res_cf)
        
    return outlist, outlist2, outlist_cf


if __name__ == "__main__":
    try:
        lang = sys.argv[1]
        assert(lang in ['fi', 'fil', 'sv', 'svl', 'svk'])
    except:
        print("usage: clustering.py [fi|fil|sv|svl|svk] <output_json_path>")
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
#        ys = ['1740', '1760', '1780', '1800', '1820', '1840', '1860', '1880', '1900']
        ys = ['1760', '1780', '1800', '1820', '1840', '1860', '1880']
        base_path =  "../models/SV_lowercase/model_sv_"
    elif lang == 'svk':
        ys = ['1740', '1760', '1780', '1800', '1820', '1840', '1860', '1880', '1900']
        base_path = "../models/SV_diachronic/models/model_sv_"
        
    prev_model = None

    res_dict = {}
    res_dict2 = {}
    res_dict_cf = {}
    
    for y in ys:
        model_path = base_path +y+".w2v"
        model = gensim.models.Word2Vec.load(model_path)

        clustering = cluster(model, prev_model)
        
        print("\n*********************\n")
        print(y)
        
        if clustering is None:
            print("nothing")
        else:
            print_cluster(*clustering)
            res_dict[y], res_dict2[y], res_dict_cf[y] = make_json(*clustering)

        prev_model = model

    jp, jp2, jp_cf = 'clustering.json', 'clustering2.json', 'clustering_cf.json'
    with open(jp, 'w') as jout:
        json.dump(res_dict, jout)
    with open(jp2, 'w') as jout:
        json.dump(res_dict2, jout)
    with open(jp_cf, 'w') as jout:
        json.dump(res_dict_cf, jout)
    
 
