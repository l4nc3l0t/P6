# %%
import os
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2 as cv

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.utils import column_or_1d


class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


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

# %%
DataImages = data[['pid', 'image']].join(CategoryTree.category_0)
# %%
# Nb d'images par cat
DataImages.groupby('category_0').agg({'image': 'count'})
# %%
# test ouverture image
fig = px.imshow(cv.imread(DataImages.path[0]))
fig.show(renderer='notebook')
# %%
DataImages['height'] = [cv.imread(p).shape[0] for p in DataImages.path]
DataImages['width'] = [cv.imread(p).shape[1] for p in DataImages.path]
# %% [markdown]
##### Visualisation des traitements de l'image
# %%
img = cv.imread(DataImages.path[0])
imgR = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
fig = px.imshow(imgR)
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show(renderer='jpeg')
# %%
def figVisuImg(img):
    fig = make_subplots(
        1,
        3,
        subplot_titles=('Image',
                        'Histogramme des pixels<br>par niveaux de gris',
                        'Histogramme des pixels<br>par niveaux de gris cumulés'),
        horizontal_spacing=.12)
    fig.add_trace(
        go.Heatmap(z=imgG, colorscale='gray', showscale=False, name='Image'))
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, autorange='reversed', row=1, col=1)
    fig.add_trace(
        go.Histogram(x=img.flatten(),
                    ybins={'size': 1},
                    name='NivGris',
                    showlegend=False), 1, 2)
    fig.add_trace(
        go.Histogram(x=img.flatten(),
                    ybins={'size': 1},
                    cumulative_enabled=True,
                    name='CumNivGris',
                    showlegend=False), 1, 3)
    fig.update_yaxes(title_text='Nombre de pixels', row=1, col=2)
    fig.update_yaxes(title_text='Nombre de pixels cumulés', row=1, col=3)
    fig.update_xaxes(title_text='Niveaux de gris', row=1, col=2)
    fig.update_xaxes(title_text='Niveaux de gris', row=1, col=3)
    fig.update_layout(height=350, width=800)
    return fig
# %%
imgG = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
fig = figVisuImg(imgG)
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgGrey.pdf')

# %%
imgGFilt = cv.GaussianBlur(imgG, (3, 3), cv.BORDER_DEFAULT)
fig = figVisuImg(imgGFilt)
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgGFilt.pdf')
# %%
imgGFiltEQ = cv.equalizeHist(imgGFilt)
fig = figVisuImg(imgGFiltEQ)
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgGFiltEQ.pdf')
# %%
