# %%
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import nltk

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
ProdSpecCleanFill = ProdSpecClean.fillna('unknown')
ProdSpecCleanFill.head()
# %%
# utilisation des 3 premières branches de catégories
TextData = CategoryTree.iloc[:, :3].merge(
    ProdSpecCleanFill, left_index=True,
    right_index=True).merge(data[['pid', 'description']], left_index=True,
                            right_index=True).set_index('pid').reset_index()
# %%
# liste des identifiants produits et des fichiers d'image associés
ImgList = data[['pid', 'image']]
# %%
