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
stopW = []
stopW = nltk.corpus.stopwords.words('english')
stopW.append([char for char in string.ascii_lowercase])  # supp lettres uniques
TokensStopW = {}
for r in range(len(TextData)):
    TokensStopW[TextData.pid[r]] = [
        word for word in Tokens[TextData.pid[r]] if not word.isdigit()
        if word not in stopW
    ]

# %% [markdown]
# Visualisation des mots ayant le plus d'occurences dans tout le jeux de données
# %%
FullTok = []
for r in range(len(TextData)):
    for tok in [*TokensStopW[TextData.pid[r]]]:
        FullTok.append(tok)
# %%
FreqTokFull = pd.DataFrame({
    'Mots': nltk.FreqDist(FullTok).keys(),
    'Freq': nltk.FreqDist(FullTok).values()
})
FreqTokFull['Freq_%'] = round(FreqTokFull.Freq * 100 / FreqTokFull.Freq.sum(),
                              2)
FreqTokFull.sort_values(['Freq'], ascending=False).head(20)

# %%
# 50 premiers mots
fig = px.bar(FreqTokFull.sort_values(['Freq'], ascending=False).head(50),
             x='Mots',
             y='Freq',
             width=900,
             labels={
                 'Freq': "Nb d'occurences",
             })
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqTok50.pdf')
    fig.write_image('./Figures/FreqTok50.pdf')
# %% [markdown]
# Nous allons retirer les mots ayant plus de 400 occurences (>0,8%)
#  car on observe une certaine rupture à ce palier et ces mots concernent
# l'aspect commercial et non le produit lui même
# %%
if write_data is True:
    FreqTokFull[FreqTokFull['Freq_%'] > 0.8].Mots.to_latex(
        './Tableaux/Mots400+.tex', index=False)
Mots400 = FreqTokFull[FreqTokFull['Freq_%'] > 0.8].sort_values(
    ['Freq'], ascending=False).Mots.to_list()
print(Mots400)
# %%
stopWtok = []
stopWtok = stopW
for word in Mots400:
    stopWtok.append(word)
TokensClean = {}
for r in range(len(TextData)):
    TokensClean[TextData.pid[r]] = [
        word for word in TokensStopW[TextData.pid[r]] if word not in stopWtok
    ]
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
FullLem = []
for r in range(len(TextData)):
    for lem in [*Lems[TextData.pid[r]]]:
        FullLem.append(lem)
# %%
FreqLemFull = pd.DataFrame({
    'Lemmes': nltk.FreqDist(FullLem).keys(),
    'Freq': nltk.FreqDist(FullLem).values()
})
FreqLemFull['Freq_%'] = round(FreqLemFull.Freq * 100 / FreqLemFull.Freq.sum(),
                              2)
FreqLemFull.sort_values(['Freq'], ascending=False).head(20)

# %%
# 50 premiers mots
fig = px.bar(FreqLemFull.sort_values(['Freq'], ascending=False).head(50),
             x='Lemmes',
             y='Freq',
             width=900,
             labels={
                 'Freq': "Nb d'occurences",
             })
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqLem50.pdf')
    fig.write_image('./Figures/FreqLem50.pdf')

# %% [markdown]
# Nous allons supprimer le lemme ayant plus de 500 occurences (>1%)
# %%
if write_data is True:
    FreqLemFull[FreqLemFull['Freq_%'] > 1].Lemmes.to_latex(
        './Tableaux/Lemmes1+.tex', index=False)
Lemmes1 = FreqLemFull[FreqLemFull['Freq_%'] > 1].sort_values(
    ['Freq'], ascending=False).Lemmes.to_list()
print(Lemmes1)
# %%
stopWlem = []
stopWlem = stopWtok
for word in Lemmes1:
    stopWlem.append(word)
LemsClean = {}
for r in range(len(TextData)):
    LemsClean[TextData.pid[r]] = [
        word for word in Lems[TextData.pid[r]] if word not in stopWlem
    ]

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
FullStem = []
for r in range(len(TextData)):
    for stem in [*Stems[TextData.pid[r]]]:
        FullStem.append(stem)
# %%
FreqStemFull = pd.DataFrame({
    'Stemmes': nltk.FreqDist(FullStem).keys(),
    'Freq': nltk.FreqDist(FullStem).values()
})
FreqStemFull['Freq_%'] = round(
    FreqStemFull.Freq * 100 / FreqStemFull.Freq.sum(), 2)
FreqStemFull.sort_values(['Freq'], ascending=False).head(20)

# %%
# 50 premiers mots
fig = px.bar(FreqStemFull.sort_values(['Freq'], ascending=False).head(50),
             x='Stemmes',
             y='Freq',
             width=900,
             labels={
                 'Freq': "Nb d'occurences",
             })
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/FreqStem50.pdf')
    fig.write_image('./Figures/FreqStem50.pdf')

# %% [markdown]
# Nous allons supprimer le stemme ayant plus de 500 occurences (>1%)
# %%
if write_data is True:
    FreqStemFull[FreqStemFull['Freq_%'] > 1].Stemmes.to_latex(
        './Tableaux/Stemmes1+.tex', index=False)
Stemmes1 = FreqStemFull[FreqStemFull['Freq_%'] > 1].sort_values(
    ['Freq'], ascending=False).Stemmes.to_list()
print(Stemmes1)
# %%
stopWstem = []
stopWstem = stopWtok
for word in Stemmes1:
    stopWstem.append(word)
StemsClean = {}
for r in range(len(TextData)):
    StemsClean[TextData.pid[r]] = [
        word for word in Stems[TextData.pid[r]] if word not in stopWstem
    ]
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
