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

# %%
DataImages = data[['pid', 'image']].join(CategoryTree.category_0)
# %%
# Nb d'images par cat
DataImages.groupby('category_0').agg({'image': 'count'})
# %%
DataImages['path'] = ['./Images/' + name for name in DataImages.image]
# %%
# test ouverture image
fig = px.imshow(cv.imread(DataImages.path[0]))
fig.show(renderer='notebook')
# %%
DataImages['height'] = [cv.imread(p).shape[0] for p in DataImages.path]
DataImages['width'] = [cv.imread(p).shape[1] for p in DataImages.path]
# %% [markdown]
##### Visualisation des traitements de l'image
# %%
img = cv.imread(DataImages.path[0])
imgR = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
fig = px.imshow(imgR)
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show(renderer='jpeg')


# %%
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
# filtration GaussianBlur
imgBWEQGaussFilt = cv.GaussianBlur(imgBWEQ, (3, 3), cv.BORDER_DEFAULT)
fig = figVisuImg(imgBWEQGaussFilt)
fig.update_layout(
    title_text='Image égalisée par equalizeHist et filtrée par GaussianBlur')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWEQGaussFilt.pdf')

# %%
# filtration Non-local Means Denoising
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
# %%
# filtration NlMD
imgBWCLAHENlMDFilt = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
fig = figVisuImg(imgBWCLAHENlMDFilt)
fig.update_layout(
    title_text=
    'Image égalisée par CLAHE et filtrée par Non-local Means Denoising')
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgBWCLAHENlMDFilt.pdf')


# %% [markdown]
# La combinaison de l'égalisation par CLAHE et de la filtration par NlMD semble
# nous retourner une image bien équilibrée (courbe cumulée régulière).
# Nous utiliserons ces traitements pour l'ensemble des images
# %% [markdown]
#### Création des images traitées
def ImgPreprocessing(imgPath, imgName):
    ImagesPreproc = {}
    if write_data is True:
        try:
            os.mkdir("./ImagesProcessed/")
        except OSError as error:
            print(error)
    for path, name in zip(imgPath, imgName):
        img = cv.imread(path)
        imgR = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
        imgBW = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        imgBWCLAHE = cv.createCLAHE(clipLimit=8,
                                    tileGridSize=(3, 3)).apply(imgBW)
        imgBWCLAHENlMD = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
        if write_data is True:
            cv.imwrite('./ImagesProcessed/' + name, imgBWCLAHENlMD)
        ImagesPreproc[name] = imgBWCLAHENlMD
    return (ImagesPreproc)


# %%
ImagesPreproc = ImgPreprocessing(DataImages.path, DataImages.image)
# %% [markdown]
#### SIFT
# %%
img = ImagesPreproc[DataImages.image[0]]
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(img, None)
imgKP = cv.drawKeypoints(img,
                         kp,
                         None,
                         flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fig = px.imshow(imgKP)
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show(renderer='jpeg')
if write_data is True:
    fig.write_image('./Figures/imgKP.pdf')
# %%
Descriptors = {}
BoVW = []
for key, value in ImagesPreproc.items():
    kp, des = sift.detectAndCompute(value, None)
    Descriptors[key] = des
    if len(BoVW) == 0:
        BoVW = des
    else:
        BoVW = np.vstack((BoVW, des))
print(BoVW.shape)
# %%
idx = []
for i in DataImages.image:
    idx.extend(len(Descriptors[i]) * [i])
BoVWDF = pd.DataFrame(
    BoVW, index=idx).reset_index().rename(columns={'index': 'ImgName'})
# %%
# Clustering
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.01)
flags = cv.KMEANS_PP_CENTERS
compactness, labels, centers = cv.kmeans(BoVW, 1000, None, criteria, 3, flags)

# %%
Lab = pd.DataFrame(
    labels.ravel(), index=idx,
    columns=['label']).reset_index().rename(columns={'index': 'ImgName'})
# %%
LabClean = Lab.groupby('ImgName').value_counts().reset_index().pivot(
    index='ImgName', columns='label', values=0).fillna(0)
LabClean.columns.name = None
LabClean.head(5)
# %%
# histogramme des descripteurs
fig = px.bar(LabClean.iloc[0],
             x=LabClean.iloc[0].index,
             y=LabClean.iloc[0].values,
             labels={
                 'index': 'Visual word',
                 'y': 'Fréquence'
             },
             width=1000,
             height=300,
             title='Histogramme des visuals words')
fig.show(renderer='notebook')
# %%
pca = PCA(n_components=.90, random_state=0)
LabPCA = pca.fit_transform(LabClean)
print('Réduction de dimensions : {} vs {}'.format(pca.n_components_,
                                                  pca.n_features_))
# %%
tsneLab = TSNE(n_components=2,
               learning_rate='auto',
               random_state=0,
               init='pca',
               n_jobs=-1).fit_transform(
                   LabPCA)
# %%
cat = []
for i in LabClean.index:
    cat.extend(DataImages[DataImages.image == i].category_0.to_list())
tsnefig = px.scatter(tsneLab, x=0, y=1, color=cat)
tsnefig.update_traces(marker_size=4)
tsnefig.update_layout(legend={'itemsizing': 'constant'})
tsnefig.show(renderer='notebook')
# %%
