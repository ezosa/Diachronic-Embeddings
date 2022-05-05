# Diachronic-Embeddings

This is code that was used in our recent work on clustering of diachronic word embeddings: 


Marjanen, J., Kurunmäki, J. A., Pivovarova, L., & Zosa, E. (2020). The expansion of isms, 1820-1917: Data-driven analysis of political language in digitized newspaper collections. Journal of Data Mining and Digital Humanities. <https://doi.org/10.46298/jdmdh.6159>


Pivovarova, L., Marjanen, J., & Zosa, E. (2019). Word Clustering for Historical Newspapers Analysis. In Ranlp Workshop on Language technology for Digital Humanities.

Marjanen, J., Pivovarova, L., Zosa, E., & Kurunmäki, J. (2019). Clustering ideological terms in historical newspaper data with diachronic word embeddings. In 5th International Workshop on Computational History, HistoInformatics 2019. CEUR-WS.

## Models

Diachronic embeddings built on the National Library of Finland newspaper collection could be downloaded from [here](https://zenodo.org/record/3557480#.XeEiPXUzYUE).

We used an incremental training method, closely following (Kim et al., 2014) and previously applied by (Hengchen et. al, 2019). More explanations, code and several embeddings model check could be found [here](https://zenodo.org/record/3270648#.XeEbMHUzYUE).


## Clustering

Once you obtained embeddings you can apply clustering using ```clustering.py``` (clustering of selected words) or ```cluster_all.py``` (enriched clustering).

Currently the code uses hard-coded links to models and hardcoded list of words.


## Vizualization

The clustering outputs json files that can be used to make Sankey chart using ```diachronic_shift_sankey.py```. 

Selected embeddings could be also vizualized using ```embeddings_drift_tsne.py```.

The code currently uses hard-coded paths.

