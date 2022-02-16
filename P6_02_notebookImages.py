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
DataImages['path'] = ['./Images/' + name for name in DataImages.image]
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
        subplot_titles=(
            'Image', 'Histogramme des pixels<br>par niveaux de gris',
            'Histogramme des pixels<br>par niveaux de gris cumulés'),
        horizontal_spacing=.12)
    fig.add_trace(
        go.Heatmap(z=img, colorscale='gray', showscale=False, name='Image'))
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
imgBW = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
fig = figVisuImg(imgBW)
fig.update_layout(title_text='Image convertie en niveaux de gris')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBW.pdf')
# %%
# égalisation par equalizeHist
imgBWEQ = cv.equalizeHist(imgBW)
fig = figVisuImg(imgBWEQ)
fig.update_layout(title_text='Image égalisée par equalizeHist')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWEQ.pdf')
# %%
# filtration GaussianBlur
imgBWEQGaussFilt = cv.GaussianBlur(imgBWEQ, (3, 3), cv.BORDER_DEFAULT)
fig = figVisuImg(imgBWEQGaussFilt)
fig.update_layout(
    title_text='Image égalisée par equalizeHist et filtrée par GaussianBlur')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWEQGaussFilt.pdf')

# %%
# filtration Non-local Means Denoising
imgBWEQNlMDFilt = cv.fastNlMeansDenoising(imgBWEQ, None, 5, 7, 21)
fig = figVisuImg(imgBWEQNlMDFilt)
fig.update_layout(
    title_text=
    'Image égalisée par equalizeHist et filtrée par Non-local Means Denoising')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWEQNlMDFilt.pdf')

# %%
# égalisation par CLAHE
imgBWCLAHE = cv.createCLAHE(clipLimit=8, tileGridSize=(3, 3)).apply(imgBW)
fig = figVisuImg(imgBWCLAHE)
fig.update_layout(title_text='Image égalisée par CLAHE')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWCLAHE.pdf')
# %%
# filtration GaussianBlur
imgBWCLAHEGaussFilt = cv.GaussianBlur(imgBWCLAHE, (3, 3), cv.BORDER_DEFAULT)
fig = figVisuImg(imgBWCLAHEGaussFilt)
fig.update_layout(
    title_text='Image égalisée par CLAHE et filtrée par GaussianBlur')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWCLAHEGaussFilt.pdf')
# %%
# filtration NlMD
imgBWCLAHENlMDFilt = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
fig = figVisuImg(imgBWCLAHENlMDFilt)
fig.update_layout(
    title_text=
    'Image égalisée par CLAHE et filtrée par Non-local Means Denoising')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWCLAHENlMDFilt.pdf')
# %% [markdown]
# La combinaison de l'égalisation par CLAHE et de la filtration par NlMD semble
# nous retourner une image bien équilibrée (courbe cumulée régulière).
# Nous utiliserons ces traitements pour l'ensemble des images
# %% [markdown]
#### Création des images traitées
# %%
if write_data is True:
    try:
        os.mkdir("./ImagesProcessed/")
    except OSError as error:
        print(error)
    for path, name in zip(DataImages.path, DataImages.image):
        img = cv.imread(path)
        imgR = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
        imgBW = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        imgBWCLAHE = cv.createCLAHE(clipLimit=8,
                                    tileGridSize=(3, 3)).apply(imgBW)
        imgBWCLAHENlMD = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
        cv.imwrite('./ImagesProcessed/' + name, imgBWCLAHENlMD)
else:
    print("""Autoriser l'écriture des fichier en changeant write_data=True""")
# %%
