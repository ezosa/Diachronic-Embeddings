from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import collections as mc


model_years = ['1860','1880','1900']

#words_path = "FI_out/ism_fi_reverse.txt"
#words_data = open(words_path,'rb').read().decode('utf-8').split("\n")
#words = [w.split() for w in words_data]
#words = [w[0] for w in words if len(w)>0 and len(w[0])>4]
#ism_words = list(set(words))

# words_path = 'SV_out/korp.json'
# words_json = json.load(open(words_path, 'rb'))
# corp = words_json['corpora']
# keys = list(corp.keys())
# words = []
# for k in keys:
#     w = list(corp[k]['relative'].keys())
#     words.extend(w)
# words = [w.lower() for w in words if len(w) > 3]
# words = [w.replace('\x84', '') for w in words]
# ism_words = list(set(words))

# common_words = []
# print("Getting words common to all time slices")
# for year_index in range(len(model_years)):
#     print("Year: ", model_years[year_index])
#     path = 'FI_out/model_fi_'+model_years[year_index]+'.w2v'
#     model = Word2Vec.load(path)
#     vocab = list(model.wv.vocab.keys())
#     ism_words_in_year = [w for w in ism_words if w in vocab]
#     common_words.append(ism_words_in_year)
#
# target_words = common_words[0]
# for i in range(1,len(model_years)-1):
#     target_words = set(target_words).intersection(set(common_words[i]))
# target_words = list(set(target_words))

#target_words = ['liberalismen', 'liberalism', 'fanatism', 'fanatismen', 'despotism', 'magnetismen', 'patriotism', 'egoism', 'despotismens', 'despotismen']
#target_words = ['modern','finsk','konduktören']
#target_words = ['kronstadt','kartio']

target_words = ['flygare', 'finska', 'patriotism']
main_path = "/home/local/pivovaro/NewsEye/diachronic_embeddings/models/SV_lowercase/model_sv_"

#plot each target word across all timeslices
print("target words: ", len(target_words))
print(target_words)
print("\nPlotting each target word...")
for target in target_words:
    print("Target word: ", target)
    target_vectors = {}
    for year_index in range(len(model_years)):
        print("Year: ", model_years[year_index])
        path = 'FI_out/model_fi_' + model_years[year_index] + '.w2v'
        path = main_path + model_years[year_index] + '.w2v'
        model = Word2Vec.load(path)
        vocab = list(model.wv.vocab.keys())
        target_word_year = target+"_"+model_years[year_index]
        target_vectors[target_word_year] = {}
        target_vectors[target_word_year]['vector'] = model[target]
        target_vectors[target_word_year]['type'] = 'target_word'
        target_word_vec = [model[target]]
        vocab_sim = [cosine_similarity(target_word_vec, [model[vocab_word]]) for vocab_word in vocab if vocab_word != target]
        word_sim = [(w,s) for s,w in sorted(zip(vocab_sim, vocab), reverse=True)][:40]
        for ws in word_sim:
            if (ws[0] not in target_vectors.keys()) or (ws[0] in target_vectors.keys() and ws[1]>target_vectors[ws[0]]['sim']):
                target_vectors[ws[0]] = {}
                target_vectors[ws[0]]['vector'] = model[ws[0]]
                target_vectors[ws[0]]['type'] = model_years[year_index]
                target_vectors[ws[0]]['sim'] = ws[1]
    words_to_plot = list(target_vectors.keys())
    len_words = len(words_to_plot)
    if len_words>2:
        print("words to plot:", len_words)
        print(words_to_plot)
        vectors = [target_vectors[w]['vector'] for w in words_to_plot]
        word_types = [target_vectors[w]['type'] for w in words_to_plot]
        df = {}
        df['words'] = words_to_plot
        df['type'] = word_types
        df = pd.DataFrame.from_dict(df)
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate=100.0)
        tsne_results = tsne.fit_transform(list(vectors))
        print('t-SNE done!')
        df['tsne-one'] = tsne_results[:,0]
        df['tsne-two'] = tsne_results[:,1]
        plt.clf()
        plt.figure(figsize=(16,10))
        # scatterplot the two principal components
        n_colors = len(list(set(df['type'])))
        #print("colors: ", n_colors)
        #print(df)
        ax = sns.scatterplot(
            x='tsne-one', y='tsne-two',
            hue='type', s=50,
            palette=sns.color_palette("hls", n_colors),
            data=df,
            legend='full',
            alpha=1.0
        )
        # label points with words
        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                if i%2 == 0:
                    ax.text(point['x']+.02, point['y'], str(point['val']))
                else:
                    ax.text(point['x'] + .02, point['y'] - .02, str(point['val']))
        label_point(df['tsne-one'], df['tsne-two'], df['words'], plt.gca())
        # draw lines between target words from different time points
        df_target = df[df['type']=='target_word']
        nrows = df_target.shape[0]
        lines = []
        for row_num in range(nrows-1):
            # draw line from time t to time t+1
            row1 = df_target.iloc[row_num,:]
            row2 = df_target.iloc[row_num+1,:]
            p1 = (row1['tsne-one'], row1['tsne-two'])
            p2 = (row2['tsne-one'], row2['tsne-two'])
            lines.append([p1,p2])
        lc = mc.LineCollection(lines, linewidths=1)
        ax.add_collection(lc)
        fig = ax.get_figure()
#        fig.savefig('FI_out/plots/tsne_'+target+'.png')
        fig.savefig('tsne_'+target+'.png')
        plt.close()
