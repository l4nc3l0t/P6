# %%
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import string
import nltk

nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'],
              '.env/lib/nltk_data')

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
    ProdSpecCleanFill, left_index=True,
    right_index=True).merge(data[['pid', 'description']],
                            left_index=True,
                            right_index=True).set_index('pid').reset_index()
# %%
# liste des identifiants produits et des fichiers d'image associés
ImgList = data[['pid', 'image']]


# %%
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
# nettoyage stopwords
stopW = nltk.corpus.stopwords.words('english')
stopW.append([char for char in string.ascii_lowercase])  # supp lettres uniques
TokensClean = {}
for r in range(len(TextData)):
    TokensClean[TextData.pid[r]] = [
        word for word in Tokens[TextData.pid[r]] if not word.isdigit()
        if word not in stopW
    ]

# %%
# lemmatisation
Lems = {}
for r in range(len(TextData)):
    Lems[TextData.pid[r]] = [
        nltk.WordNetLemmatizer().lemmatize(word)
        for word in TokensClean[TextData.pid[r]]
    ]
# %%
FullTok = []
for r in range(len(TextData)):
    for word in [*TokensClean[TextData.pid[r]]]:
        FullTok.append(word)
# %%
FreqFull = pd.DataFrame({
    'Mot': nltk.FreqDist(FullTok).keys(),
    'Freq': nltk.FreqDist(FullTok).values()
})
FreqFull['Freq_%'] = round(FreqFull.Freq * 100 / FreqFull.Freq.count(), 2)
FreqFull.sort_values(['Freq'], ascending=False).head(20)

# %%
fig = px.bar(FreqFull.sort_values(['Freq'], ascending=False).head(50),
             x='Mot',
             y='Freq',
             width=1000)
fig.show(renderer='notebook')
# %%
