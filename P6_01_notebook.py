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
CategoryTree.info()
# %% [markdown]
# Les 3 premières hiérarchies de catégories sont bien remplies les suivantes à moitié

# %% [markdown]
# nettoyage des spécifications produits et mise sous forme de dataframe
# %%
# nettoyage
Specifications = data.product_specifications.str.replace(
    '"product_specification"=>', '',
    regex=True).str.replace('{"key"=>', '', regex=True).str.replace(
        ', "value"=>', ':',
        regex=True).str.replace('},', ',', regex=True).str.replace(
            '[', '', regex=True).str.replace(']}', '', regex=True)
Specifications.head(3)


# %%
def try_the_eval(row):
    try:
        ast.literal_eval(row)
    except:
        print('Found bad data: {0}'.format(row))


evalError = Specifications.apply(try_the_eval)
# %%
SpecificationsClean = Specifications.str.replace('{"value"=>',
                                                 '"unknown":',
                                                 regex=True)

# %%
evalError = SpecificationsClean.apply(try_the_eval)
# %%
SpecificationsClean = SpecificationsClean.str.replace('{nil}',
                                                      '{"unknown":"unknown"}',
                                                      regex=True).str.replace(
                                                          '}}',
                                                          '}',
                                                          regex=True)
SpecificationsClean.fillna('{"unknown":"unknown"}', inplace=True)
# %%
# mise sous forme de dictionnaire puis de dataframe
for i in SpecificationsClean.index:
    spec = ast.literal_eval(SpecificationsClean[i])
    if i == 0:
        ProdSpec = pd.DataFrame(columns=spec.keys())
        ProdSpec = ProdSpec.merge(pd.DataFrame(spec, index=[i]),
                                  on=ProdSpec.columns[ProdSpec.columns.isin(
                                      spec.keys())].to_list(),
                                  how='outer')
    else:
        ProdSpec = ProdSpec.merge(pd.DataFrame(spec, index=[i]),
                                  on=ProdSpec.columns[ProdSpec.columns.isin(
                                      spec.keys())].to_list(),
                                  how='outer')
# %%
ProdSpec.head()
# %%
# visualisation des colonnes comportant le plus de données
ProdSpec.isna().sum().sort_values().head(15)
# %%
# conservation des 7 colonnes contenant le plus de données
ProdSpecClean = ProdSpec[ProdSpec.isna().sum().sort_values().head(
    8).index.drop('unknown').to_list()]
ProdSpecClean.head()
#ProdSpecCleanFill = ProdSpecClean.fillna('unknown')
#ProdSpecCleanFill.head()
# %%
# utilisation des 3 premières branches de catégories
TextData = CategoryTree.iloc[:, :3].merge(
    ProdSpecClean, left_index=True,
    right_index=True).merge(data[['pid', 'description']],
                            left_index=True,
                            right_index=True).set_index('pid').reset_index()
# %%
# liste des identifiants produits et des fichiers d'image associés
ImgList = data[['pid', 'image']]


# %%
# nettoyage ponctuation, nombres
def clean_text(text):
    textNpunct = ''.join(
        [char for char in text if char not in string.punctuation])
    textNnum = ''.join([char for char in textNpunct if not char.isdigit()])
    textClean = textNnum
    return textClean


Descriptions = TextData.description.apply(lambda x: clean_text(x))
# %%
# tokenisation
Tokens = {}
for r in range(len(TextData)):
    Tokens[TextData.pid[r]] = nltk.word_tokenize(Descriptions.loc[r].lower())


# %%
# nettoyage stopwords
def cleanStopW(List, stopW=[], addtolist=[], index=TextData.pid):
    for atl in addtolist:
        stopW.append(atl)
    ListClean = {}
    for r in range(len(index)):
        ListClean[index[r]] = [
            word for word in List[index[r]] if not word.isdigit()
            if word not in stopW
        ]
    return (ListClean)


# %%
stopW = []
stopW = nltk.corpus.stopwords.words('english')
TokensStopW = cleanStopW(Tokens, stopW, string.ascii_lowercase)


# %% [markdown]
# Visualisation des mots ayant le plus d'occurences dans tout le jeux de données
# %%
def visuWordList(Words, listname='Tokens'):
    FullW = []
    for r in range(len(TextData)):
        for item in [*Words[TextData.pid[r]]]:
            FullW.append(item)
    FreqWFull = pd.DataFrame({
        listname: nltk.FreqDist(FullW).keys(),
        'Freq': nltk.FreqDist(FullW).values()
    })
    FreqWFull['Freq_%'] = round(FreqWFull.Freq * 100 / FreqWFull.Freq.sum(), 2)
    print(FreqWFull.sort_values(['Freq'], ascending=False).head(20))

    fig = px.bar(FreqWFull.sort_values(['Freq'], ascending=False).head(50),
                 x=listname,
                 y='Freq',
                 width=900,
                 labels={
                     'Freq': "Nb d'occurences",
                 })
    return (FullW, FreqWFull, fig)


# %%
FullTok, FreqTokFull, fig = visuWordList(TokensStopW)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqTok50.pdf')
    fig.write_image('./Figures/FreqTok50.pdf')

# %% [markdown]
# Nous allons retirer les mots ayant plus de 400 occurences (> 0,8%)
# car on observe une certaine rupture à ce palier et ces mots concernent
# l'aspect commercial et non le produit lui même. Nous allons conserverons
# que les mots ayant au moins 3 occurences (>= 0.01%)
# %%
if write_data is True:
    FreqTokFull[FreqTokFull['Freq'] > 400].Tokens.to_latex(
        './Tableaux/Mots400+.tex', index=False)
FiltMots = FreqTokFull[(FreqTokFull['Freq'] > 400) |
                       (FreqTokFull['Freq'] < 3)].Tokens.to_list()
# %%
TokensClean = cleanStopW(TokensStopW, stopW, FiltMots)
# %%
# lemmatisation
Lems = {}
for r in range(len(TextData)):
    Lems[TextData.pid[r]] = [
        nltk.WordNetLemmatizer().lemmatize(word)
        for word in TokensClean[TextData.pid[r]]
    ]
# %% [markdown]
# Visualisation des lemmes ayant le plus d'occurences dans tout le jeux de données
# %%
FullLem, FreqLemFull, fig = visuWordList(Lems, 'Lemmes')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqLem50.pdf')

# %% [markdown]
# Nous allons supprimer le lemme ayant plus de 500 occurences (> 1.3%)
# %%
if write_data is True:
    FreqLemFull[FreqLemFull['Freq'] > 500].Lemmes.to_latex(
        './Tableaux/Lemmes1+.tex', index=False)
Lemmes1 = FreqLemFull[FreqLemFull['Freq'] > 500].sort_values(
    ['Freq'], ascending=False).Lemmes.to_list()
print(Lemmes1)
# %%
LemsClean = cleanStopW(Lems, stopW, FiltMots + Lemmes1)
# %%
# racinisation (stemming)
Stems = {}
for r in range(len(TextData)):
    Stems[TextData.pid[r]] = [
        nltk.stem.PorterStemmer().stem(word)
        for word in TokensClean[TextData.pid[r]]
    ]
# %% [markdown]
# Visualisation des racines ayant le plus d'occurences dans tout le jeux de données
# %%
FullStem, FreqStemFull, fig = visuWordList(Stems, 'Racines')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqStem50.pdf')
# %% [markdown]
# Nous allons supprimer le stemme ayant plus de 500 occurences (> 1.3%)
# %%
if write_data is True:
    FreqStemFull[FreqStemFull['Freq'] > 500].Racines.to_latex(
        './Tableaux/Stemmes1+.tex', index=False)
Stemmes1 = FreqStemFull[FreqStemFull['Freq'] > 500].sort_values(
    ['Freq'], ascending=False).Racines.to_list()
print(Stemmes1)
# %%
StemsClean = cleanStopW(Stems, stopW, FiltMots + Stemmes1)
# %%
# Comparaison texte brute, tokens, lemmatisation, racinisation
CompareTxt = pd.DataFrame({
    'Modification':
    ['Texte brut', 'Tokenisation', 'Lemmatisation', 'Racinisation'],
    'Contenu': [
        TextData.description[0],
        ' '.join(tok for tok in TokensClean[TextData.pid[0]]),
        ' '.join(lem for lem in LemsClean[TextData.pid[0]]),
        ' '.join(stem for stem in StemsClean[TextData.pid[0]])
    ]
})
if write_data is True:
    CompareTxt.to_latex('./Tableaux/CompareTxt.tex', index=False)
CompareTxt
# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import column_or_1d


class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


# %%
corporaTok = []
for r in range(len(TextData)):
    corporaTok.append(' '.join(tok for tok in TokensClean[TextData.pid[r]]))


# %%
def clustering(corpora, vectorizer=[TfidfVectorizer()], TokenType=None):
    Labels = {}

    color_discrete_map = {}
    category_orders = TextData.category_0.sort_values().unique()
    perplexity = [10, 20, 30, 40, 50]
    for cat, col in zip(TextData.category_0.unique(),
                        px.colors.qualitative.D3[0:6]):
        color_discrete_map[cat] = col

    Scores = pd.DataFrame(columns=['Vectorizer', 'perplexityTSNE', 'ARI'])
    i = 0

    for p in perplexity:
        for vec in vectorizer:
            if str(vec) == str(HashingVectorizer()):
                cv = CountVectorizer()
                vec = HashingVectorizer(
                    n_features=cv.fit_transform(corpora).get_shape()[1])
                vectorizedTok = vec.fit_transform(corpora)
                vectorizedTokDF = pd.DataFrame(vectorizedTok.toarray(),
                                               TextData.pid,
                                               cv.get_feature_names_out())
            else:
                vectorizedTok = vec.fit_transform(corpora)
                vectorizedTokDF = pd.DataFrame(vectorizedTok.toarray(),
                                               TextData.pid,
                                               vec.get_feature_names_out())

            pca = PCA(n_components=.98, random_state=0)
            vecTokPCA = pca.fit_transform(vectorizedTokDF)
            print('Réduction de dimensions : {} vs {}'.format(
                pca.n_components_, pca.n_features_))

            tsneVTok = TSNE(n_components=2,
                            perplexity=p,
                            learning_rate='auto',
                            random_state=0,
                            init='pca',
                            n_jobs=-1).fit_transform(vecTokPCA)

            tsnefig = px.scatter(tsneVTok,
                                 x=0,
                                 y=1,
                                 color=TextData.category_0,
                                 color_discrete_map=color_discrete_map,
                                 category_orders={'color': category_orders},
                                 labels={'color': 'Catégories'},
                                 opacity=1,
                                 title='t-SNE{} {} {}'.format(
                                     p,
                                     str(vec).split('(')[0], TokenType))
            tsnefig.update_traces(marker_size=4)
            tsnefig.update_layout(legend={'itemsizing': 'constant'})
            tsnefig.show(renderer='notebook')
            if write_data is True:
                tsnefig.write_image('./Figures/tsne{}{}{}.pdf'.format(
                    p, TokenType,
                    str(vec).split('(')[0]))

            vecKMeans = KMeans(n_clusters=7, random_state=0).fit(tsneVTok)

            LabelsDF = pd.DataFrame({
                'Catégories réelles': TextData.category_0,
                'Labels KMeans': vecKMeans.labels_
            })
            labelsGroups = LabelsDF.groupby(
                ['Catégories réelles'])['Labels KMeans'].value_counts()
            LabelsClean = labelsGroups.groupby(
                level=0).max().sort_values().reset_index().join(
                    pd.Series(labelsGroups.groupby(
                        level=1).max().sort_values().index.to_list(),
                              name='Label maj')).rename(
                                  columns={
                                      'Labels KMeans': 'Nb prod/label'
                                  }).sort_values('Label maj')
            print(LabelsClean)
            #print(labelsGroups)

            le = MyLabelEncoder()
            le.fit(LabelsClean['Catégories réelles'])

            LabelsDF['Labels réels'] = le.transform(
                LabelsDF['Catégories réelles'])
            LabelsDF['Catégories KMeans'] = le.inverse_transform(
                LabelsDF['Labels KMeans'])
            LabelsDF.reindex(columns=[
                'Catégories réelles', 'Labels réels', 'Labels KMeans',
                'Catégories KMeans'
            ])

            ARI = adjusted_rand_score(LabelsDF['Labels réels'],
                                      LabelsDF['Labels KMeans'])

            Scores.loc[i, 'Vectorizer'] = str(vec).split('(')[0]
            Scores.loc[i, 'perplexityTSNE'] = str(p)
            Scores.loc[i, 'ARI'] = ARI
            Scores.loc[i, 'TokenType'] = TokenType
            i += 1

            kmeansfig = px.scatter(tsneVTok,
                                   x=0,
                                   y=1,
                                   title='KMeans t-SNE{} {} {}'.format(
                                       p,
                                       str(vec).split('(')[0], TokenType),
                                   color=LabelsDF['Catégories KMeans'],
                                   color_discrete_map=color_discrete_map,
                                   category_orders={'color': category_orders},
                                   labels={'color': 'Catégories'})
            kmeansfig.update_traces(marker_size=4)
            kmeansfig.update_layout(legend={'itemsizing': 'constant'})
            kmeansfig.show(renderer='notebook')
            if write_data is True:
                kmeansfig.write_image('./Figures/kmean{}{}{}.pdf'.format(
                    p, TokenType,
                    str(vec).split('(')[0]))

            print('ARI :{}'.format(ARI))

    return (Scores)


# %%
TokScores = clustering(
    corporaTok, [TfidfVectorizer(),
                 CountVectorizer(),
                 HashingVectorizer()], 'Tokens')
# %%
print(TokScores)
# %%
corporaLem = []
for r in range(len(TextData)):
    corporaLem.append(' '.join(lem for lem in LemsClean[TextData.pid[r]]))
# %%
LemScores = clustering(
    corporaLem, [TfidfVectorizer(),
                 CountVectorizer(),
                 HashingVectorizer()], 'Lemmes')
# %%
print(LemScores)
# %%
corporaStem = []
for r in range(len(TextData)):
    corporaStem.append(' '.join(stem for stem in StemsClean[TextData.pid[r]]))
# %%
StemScores = clustering(
    corporaStem, [TfidfVectorizer(),
                  CountVectorizer(),
                  HashingVectorizer()], 'Racines')
# %%
print(StemScores)

# %%
ScoresFull = TokScores.merge(LemScores,
                             on=TokScores.columns.to_list(),
                             how='outer').merge(StemScores,
                                                on=TokScores.columns.to_list(),
                                                how='outer')
# %%
fig = px.bar(ScoresFull,
             x='perplexityTSNE',
             y='ARI',
             color='Vectorizer',
             facet_col='TokenType',
             barmode='group')
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CompareScores.pdf')
# %%