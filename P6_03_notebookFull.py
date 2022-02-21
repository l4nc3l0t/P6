# %%
import os
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import string
import nltk

nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'],
              '.env/lib/nltk_data')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.utils import column_or_1d


class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


import PIL
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
# %%
write_data = True

# True : création d'un dossier Figures et Tableau
# dans lesquels seront créés les éléments qui serviront à la présentation
# et écriture des figures et tableaux dans ces dossier
#
# False : pas de création de dossier ni de figures ni de tableaux

if write_data is True:
    try:
        os.mkdir("./Figures/")
    except OSError as error:
        print(error)
    try:
        os.mkdir("./Tableaux/")
    except OSError as error:
        print(error)
else:
    print("""Visualisation uniquement dans le notebook
    pas de création de figures ni de tableaux""")

# %%
data = pd.read_csv('./flipkart_com-ecommerce_sample_1050.csv')
# %%
data.info()
# %%
CategoryTree = data.product_category_tree.str.slice(
    start=2, stop=-2).str.split(' >> ', expand=True)
for i in CategoryTree.columns:
    CategoryTree.rename(columns={i: 'category_{}'.format(i)}, inplace=True)
CategoryTree.head(3)
# %%
# utilisation des 3 premières branches de catégories
TextData = CategoryTree.iloc[:, :1].merge(
    data[['pid', 'description']], left_index=True,
    right_index=True).set_index('pid').reset_index()
# %%
DataImages = data[['pid', 'image']].join(CategoryTree.category_0)
# %%
DataImages['path'] = ['./Images/' + name for name in DataImages.image]
# %%
DataFull = TextData.merge(DataImages[['pid', 'image', 'path']], on='pid')


# %%
# nettoyage ponctuation, nombres
def clean_text(text):
    textNpunct = ''.join(
        [char for char in text if char not in string.punctuation])
    textNnum = ''.join([char for char in textNpunct if not char.isdigit()])
    textClean = textNnum
    return textClean


Descriptions = DataFull.description.apply(lambda x: clean_text(x))
# %%
# tokenisation
Tokens = {}
for r in range(len(DataFull)):
    Tokens[DataFull.pid[r]] = nltk.word_tokenize(Descriptions.loc[r].lower())


# %%
# nettoyage stopwords
def cleanStopW(List, stopW=[], addtolist=[], index=DataFull.pid):
    for atl in addtolist:
        stopW.append(atl)
    ListClean = {}
    for r in range(len(index)):
        ListClean[index[r]] = [
            word for word in List[index[r]] if not word.isdigit()
            if word not in stopW
        ]
    return ListClean


# %%
stopW = []
stopW = nltk.corpus.stopwords.words('english')
TokensStopW = cleanStopW(Tokens, stopW, string.ascii_lowercase)


# %%
def visuWordList(Words, listname='Tokens'):
    FullW = []
    for r in range(len(DataFull)):
        for item in [*Words[DataFull.pid[r]]]:
            FullW.append(item)
    FreqWFull = pd.DataFrame({
        listname: nltk.FreqDist(FullW).keys(),
        'Freq': nltk.FreqDist(FullW).values()
    })
    FreqWFull['Freq_%'] = round(FreqWFull.Freq * 100 / FreqWFull.Freq.sum(), 2)
    print(FreqWFull.sort_values(['Freq'], ascending=False).head(20))

    return FullW, FreqWFull


# %%
FullTok, FreqTokFull = visuWordList(TokensStopW)
# %%
FiltMots = FreqTokFull[(FreqTokFull['Freq'] > 400) |
                       (FreqTokFull['Freq'] < 3)].Tokens.to_list()
# %%
TokensClean = cleanStopW(TokensStopW, stopW, FiltMots)
# %%
# lemmatisation
Lems = {}
for r in range(len(DataFull)):
    Lems[DataFull.pid[r]] = [
        nltk.WordNetLemmatizer().lemmatize(word)
        for word in TokensClean[DataFull.pid[r]]
    ]
# %%
FullLem, FreqLemFull = visuWordList(Lems, 'Lemmes')
# %%
Lemmes1 = FreqLemFull[FreqLemFull['Freq'] > 500].sort_values(
    ['Freq'], ascending=False).Lemmes.to_list()
# %%
LemsClean = cleanStopW(Lems, stopW, FiltMots + Lemmes1)
# %%
Stems = {}
for r in range(len(DataFull)):
    Stems[DataFull.pid[r]] = [
        nltk.stem.PorterStemmer().stem(word)
        for word in TokensClean[DataFull.pid[r]]
    ]
# %%
FullStem, FreqStemFull = visuWordList(Stems, 'Racines')
# %% [markdown]
# Nous allons supprimer le stemme ayant plus de 500 occurences (> 1.3%)
# %%
Stemmes1 = FreqStemFull[FreqStemFull['Freq'] > 500].sort_values(
    ['Freq'], ascending=False).Racines.to_list()
# %%
StemsClean = cleanStopW(Stems, stopW, FiltMots + Stemmes1)
# %%
# text
corporaStem = []
for r in range(len(DataFull)):
    corporaStem.append(' '.join(stem for stem in StemsClean[DataFull.pid[r]]))
vec = TfidfVectorizer(ngram_range=(1, 1))
vectorizedLem = vec.fit_transform(corporaStem)
vectorizedLemDF = pd.DataFrame(vectorizedLem.toarray(), DataFull.pid,
                               vec.get_feature_names_out())

# %%
# images
iv3 = inception_v3.InceptionV3(weights='imagenet', include_top=False)
ImagesCNN = {}
for path, name in zip(DataFull.path, DataFull.image):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    ImagesCNN[name] = x

features = []
for arr in ImagesCNN.values():
    features.append(iv3.predict(arr).flatten())
featuresA = np.array(features)

# %%
DesFull = np.concatenate((vectorizedStem.toarray(), featuresA), axis=1)
print(len(featuresA[0]) + len(vectorizedLem.toarray()[0]) == len(DesFull[0]))
# %%
DesFull_scaled = StandardScaler().fit_transform(DesFull)
Scores = pd.DataFrame(columns=['perplexityTSNE', 'ARI', 'PCAcomp'])
i = 0

for ncomp in [50, 100, 110, 120, 150]:
    pca = PCA(n_components=ncomp, random_state=0)
    DesFullPCA = pca.fit_transform(DesFull_scaled)
    print('Réduction de dimensions : {} vs {}'.format(pca.n_components_,
                                                      pca.n_features_))

    color_discrete_map = {}
    category_orders = DataFull.category_0.sort_values().unique()
    for cat, col in zip(
            DataFull.category_0.unique(),
            px.colors.qualitative.D3[0:len(DataFull.category_0.unique())]):
        color_discrete_map[cat] = col

    for p in [30, 40, 50, 60, 70]:
        tsneDes = TSNE(n_components=2,
                       perplexity=p,
                       learning_rate='auto',
                       random_state=0,
                       init='pca',
                       n_jobs=-1).fit_transform(DesFullPCA)

        tsnefig = px.scatter(tsneDes,
                             x=0,
                             y=1,
                             color=DataFull.category_0,
                             color_discrete_map=color_discrete_map,
                             category_orders={'color': category_orders},
                             labels={
                                 'color': 'Catégories',
                                 '0': 'tSNE1',
                                 '1': 'tSNE2'
                             },
                             opacity=1,
                             title='t-SNE{} Racines tfidf(1, 1) IV3 {}'.format(
                                 p, ncomp))
        tsnefig.update_traces(marker_size=4)
        tsnefig.update_layout(legend={'itemsizing': 'constant'})
        tsnefig.show(renderer='jpeg')
        if write_data is True:
            tsnefig.write_image('./Figures/tsne{}StemtfidfMonoIV3{}.pdf'.format(
                p,
                str(ncomp).replace('.', '')))

        vecKMeans = KMeans(n_clusters=7, random_state=0).fit(tsneDes)

        LabelsDF = pd.DataFrame({
            'Catégories réelles': DataFull.category_0,
            'Labels KMeans': vecKMeans.labels_
        })
        labelsGroups = LabelsDF.groupby(['Catégories réelles'
                                         ])['Labels KMeans'].value_counts()
        LabelsClean = labelsGroups.groupby(
            level=0).max().sort_values().reset_index().join(
                pd.Series(
                    labelsGroups.groupby(
                        level=1).max().sort_values().index.to_list(),
                    name='Label maj')).rename(columns={
                        'Labels KMeans': 'Nb prod/label'
                    }).sort_values('Label maj').reset_index(drop=True)
        print(LabelsClean)
        #print(labelsGroups)

        le = MyLabelEncoder()
        le.fit(LabelsClean['Catégories réelles'])

        LabelsDF['Labels réels'] = le.transform(LabelsDF['Catégories réelles'])
        LabelsDF['Catégories KMeans'] = le.inverse_transform(
            LabelsDF['Labels KMeans'])
        LabelsDF.reindex(columns=[
            'Catégories réelles', 'Labels réels', 'Labels KMeans',
            'Catégories KMeans'
        ])

        CM = confusion_matrix(LabelsDF['Catégories KMeans'],
                              LabelsDF['Catégories réelles'])
        CMfig = px.imshow(
            CM,
            x=category_orders,
            y=category_orders,
            text_auto=True,
            color_continuous_scale='balance',
            labels={
                'x': 'Catégorie prédite',
                'y': 'Catégorie réelle',
                'color': 'Nb produits'
            },
            title=
            'Matrice de confusion des labels prédits (x) et réels (y)<br>t-SNE{} Racines tfidf(1, 1) IV3 {}'
            .format(p, ncomp))
        CMfig.update_layout(plot_bgcolor='white')
        CMfig.update_coloraxes(showscale=False)
        CMfig.show(renderer='jpeg')
        if write_data is True:
            CMfig.write_image(
                './Figures/HeatmapLabels{}StemtfidfMonoIV3{}.pdf'.format(
                    p,
                    str(ncomp).replace('.', '')))

        ARI = adjusted_rand_score(LabelsDF['Labels réels'],
                                  LabelsDF['Labels KMeans'])
        Scores.loc[i, 'PCAcomp'] = str(ncomp).replace('.', '')
        Scores.loc[i, 'perplexityTSNE'] = str(p)
        Scores.loc[i, 'ARI'] = ARI
        i += 1

        kmeansfig = px.scatter(
            tsneDes,
            x=0,
            y=1,
            title='KMeans t-SNE{} Racines tfidf(1, 1) IV3 {}'.format(p, ncomp),
            color=LabelsDF['Catégories KMeans'],
            color_discrete_map=color_discrete_map,
            category_orders={'color': category_orders},
            labels={
                'color': 'Catégories',
                '0': 'tSNE1',
                '1': 'tSNE2'
            })
        kmeansfig.update_traces(marker_size=4)
        kmeansfig.update_layout(legend={'itemsizing': 'constant'})
        kmeansfig.show(renderer='jpeg')
        if write_data is True:
            kmeansfig.write_image(
                './Figures/kmean{}StemtfidfMonoIV3{}.pdf'.format(
                    p,
                    str(ncomp).replace('.', '')))

        print('ARI :{}'.format(ARI))

# %%
print(Scores)
# %%
fig = px.bar(Scores,
             x='perplexityTSNE',
             y='ARI',
             facet_col='PCAcomp',
             barmode='group',
             width=700,
             title='Comparaison des scores en fonction<br>de la perplexité')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CompareFullScores.pdf')
# %%
