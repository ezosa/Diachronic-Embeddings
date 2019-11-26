from collections import Counter
import numpy as np
import json
import random
import plotly
import plotly.plotly as py
import randomcolor
rand_color = randomcolor.RandomColor()
plotly.tools.set_credentials_file(username='', api_key='')

def random_colors(n=1):
    rc = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        rc.append('#%02X%02X%02X' % (r(),r(),r()))
    return rc


clusters = json.load(open("ism_clusters/cluster_enriched.json",'r',encoding='utf-8'))
time_slice = sorted(list(clusters.keys()))

cluster_counts = {}
source_names = []
target_names = []
for index in range(len(time_slice)-1):
    for source in clusters[time_slice[index]]:
        source_name = source[0]+"_"+time_slice[index]
        source_names.append(source_name)
        if source_name not in cluster_counts.keys():
            cluster_counts[source_name] = {}
        for target in clusters[time_slice[index + 1]]:
            target_name = target[0]+"_"+time_slice[index + 1]
            target_names.append(target_name)
            val = len(set(source).intersection(set(target)))
            cluster_counts[source_name][target_name] = val

target_names = list(set(target_names))
labels = source_names + target_names
label_dict = {labels[i]:i for i in range(len(labels))}

groups = [[] for _ in range(len(time_slice))]
for label in label_dict.keys():
    label_year = label.split("_")[1]
    label_index = time_slice.index(label_year)
    groups[label_index].append(label_dict[label])
    

color_dict = {}
for year in clusters.keys():
    for cluster in clusters[year]:
        if cluster[0] not in color_dict.keys():
            color_dict[cluster[0][:5]] = rand_color.generate()

colors = []
for label_name in labels:
    label_prefix = label_name.split("_")[0][:5]
    colors.append(color_dict[label_prefix])

source_list = []
target_list = []
val = []
for source in source_names:
    for target in cluster_counts[source].keys():
        source_list.append(source)
        target_list.append(target)
        val.append(cluster_counts[source][target])


data = dict(
    type='sankey',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(
            color="black",
            width=0.5),
        label=labels,
        color=colors,
        arrangement="freeform",
        groups=groups),
    link=dict(
      source= [label_dict[s] for s in source_list],
      target= [label_dict[t] for t in target_list],
      value= val
    )
)
layout = dict(
    title="Clusters "+time_slice[0]+"-"+time_slice[-1],
    font=dict(
      size=10
    )
)

fig = dict(data=[data], layout=layout)
py.plot(fig, validate=False)