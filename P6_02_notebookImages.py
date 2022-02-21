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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.utils import column_or_1d


class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


from PIL import Image
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
# %%
write_data = False

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
fig.show(renderer='jpeg')
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
#### Traitement des images
# %%
def ImgPreprocessing(imgPath, imgName):
    ImagesPreproc = {}
    for path, name in zip(imgPath, imgName):
        img = cv.imread(path)
        imgR = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
        imgBW = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        imgBWCLAHE = cv.createCLAHE(clipLimit=8,
                                    tileGridSize=(3, 3)).apply(imgBW)
        imgBWCLAHENlMD = cv.fastNlMeansDenoising(imgBWCLAHE, None, 5, 7, 21)
        ImagesPreproc[name] = imgBWCLAHENlMD
    return (ImagesPreproc)


# %%
ImagesPreproc = ImgPreprocessing(DataImages.path, DataImages.image)


# %% [markdown]
#### Essais SIFT, ORB et CNN
# %%
def clustering(Algo, perplexity=[30, 40, 50, 60], n_componentsPCA=100):
    Labels = {}

    color_discrete_map = {}
    category_orders = DataImages.category_0.sort_values().unique()
    for cat, col in zip(
            DataImages.category_0.unique(),
            px.colors.qualitative.D3[0:len(DataImages.category_0.unique())]):
        color_discrete_map[cat] = col

    Scores = pd.DataFrame(columns=['perplexityTSNE', 'ARI'])
    row = 0
    for a in Algo:
        if a == 'ORB' or a == 'SIFT':
            img = ImagesPreproc[DataImages.image[0]]
            if a == 'ORB':
                orb = cv.ORB_create(nfeatures=1000)
                kp, des = orb.detectAndCompute(img, None)
            if a == 'SIFT':
                sift = cv.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)

            imgKP = cv.drawKeypoints(img, kp, None)
            figKP = px.imshow(imgKP)
            figKP.update_layout(coloraxis_showscale=False)
            figKP.update_xaxes(showticklabels=False)
            figKP.update_yaxes(showticklabels=False)
            figKP.show(renderer='jpeg')
            if write_data is True:
                figKP.write_image('./Figures/{}imgKP.pdf'.format(a))

            Descriptors = {}
            BoVW = []
            for key, value in ImagesPreproc.items():
                if a == 'ORB':
                    kp, des = orb.detectAndCompute(value, None)
                if a == 'SIFT':
                    kp, des = sift.detectAndCompute(value, None)
                Descriptors[key] = des
                if len(BoVW) == 0:
                    BoVW = des
                elif des is None:
                    BoVW = np.vstack((BoVW, len(BoVW[0]) * [0]))
                else:
                    BoVW = np.vstack((BoVW, des))
            print('Dimensions du BoVW : ', BoVW.shape)
            BoVW = np.float32(BoVW)

            idx = []
            for i in DataImages.image:
                if Descriptors[i] is None:
                    idx.extend([i])
                else:
                    idx.extend(len(Descriptors[i]) * [i])
            BoVWDF = pd.DataFrame(
                BoVW,
                index=idx).reset_index().rename(columns={'index': 'ImgName'})

            # Clustering
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10,
                        1)
            flags = cv.KMEANS_PP_CENTERS
            compactness, labels, centers = cv.kmeans(BoVW, 1000, None,
                                                     criteria, 1, flags)

            Lab = pd.DataFrame(labels.ravel(), index=idx, columns=[
                'label'
            ]).reset_index().rename(columns={'index': 'ImgName'})

            LabClean = Lab.groupby(
                'ImgName').value_counts().reset_index().pivot(
                    index='ImgName', columns='label', values=0).fillna(0)
            LabClean.columns.name = None
            LabClean.head(5)

            category = []
            for i in LabClean.index:
                category.extend(
                    DataImages[DataImages.image == i].category_0.to_list())

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
                         title='Histogramme des visuals words {}'.format(a))
            fig.show(renderer='jpeg')

            LabClean_scaled = StandardScaler().fit_transform(LabClean)

        else:
            if a == 'VGG':
                vgg = vgg16.VGG16(weights='imagenet', include_top=False)
            if a == 'XEPT':
                xept = xception.Xception(weights='imagenet', include_top=False)
            if a == 'IV3':
                iv3 = inception_v3.InceptionV3(weights='imagenet',
                                               include_top=False)
            ImagesCNN = {}
            for path, name in zip(DataImages.path, DataImages.image):
                img = image.load_img(path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                if a == 'VGG':
                    x = vgg16.preprocess_input(x)
                if a == 'XEPT':
                    x = xception.preprocess_input(x)
                if a == 'IV3':
                    x = inception_v3.preprocess_input(x)
                ImagesCNN[name] = x

            features = []
            for arr in ImagesCNN.values():
                if a == 'VGG':
                    features.append(vgg.predict(arr).flatten())
                if a == 'XEPT':
                    features.append(xept.predict(arr).flatten())
                if a == 'IV3':
                    features.append(iv3.predict(arr).flatten())

            LabClean_scaled = StandardScaler().fit_transform(features)

        pca = PCA(n_components=n_componentsPCA, random_state=0)
        LabPCA = pca.fit_transform(LabClean_scaled)
        print('Réduction de dimensions : {} vs {}'.format(
            pca.n_components_, pca.n_features_))

        for p in perplexity:
            tsneLab = TSNE(n_components=2,
                           perplexity=p,
                           learning_rate='auto',
                           random_state=0,
                           init='pca',
                           n_jobs=-1).fit_transform(LabPCA)

            tsnefig = px.scatter(tsneLab,
                                 x=0,
                                 y=1,
                                 color=category if a == 'SIFT' or a == 'ORB'
                                 else DataImages.category_0,
                                 color_discrete_map=color_discrete_map,
                                 category_orders={'color': category_orders},
                                 labels={
                                     'color': 'Catégories',
                                     '0': 'tSNE1',
                                     '1': 'tSNE2'
                                 },
                                 opacity=1,
                                 title='t-SNE{} {}'.format(p, a))
            tsnefig.update_traces(marker_size=4)
            tsnefig.update_layout(legend={'itemsizing': 'constant'})
            tsnefig.show(renderer='jpeg')
            if write_data is True:
                tsnefig.write_image('./Figures/tsne{}{}.pdf'.format(p, a))

            LabKMeans = KMeans(n_clusters=7, random_state=0).fit(tsneLab)

            LabelsDF = pd.DataFrame({
                'Catégories réelles':
                category
                if a == 'SIFT' or a == 'ORB' else DataImages.category_0,
                'Labels KMeans':
                LabKMeans.labels_
            })

            labelsGroups = LabelsDF.groupby(
                ['Catégories réelles'])['Labels KMeans'].value_counts()
            LabelsClean = labelsGroups.groupby(
                level=0).max().sort_values().reset_index().join(
                    pd.Series(
                        labelsGroups.groupby(
                            level=1).max().sort_values().index.to_list(),
                        name='Label maj')).rename(
                            columns={
                                'Labels KMeans': 'Nb prod/label'
                            }).sort_values('Label maj').reset_index(drop=True)
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
                'Matrice de confusion des labels prédits (x) et réels (y)<br>t-SNE{} {}'
                .format(p, a))
            CMfig.update_layout(plot_bgcolor='white')
            CMfig.update_coloraxes(showscale=False)
            CMfig.show(renderer='jpeg')
            if write_data is True:
                CMfig.write_image('./Figures/HeatmapLabels{}{}.pdf'.format(
                    p, a))

            ARI = adjusted_rand_score(LabelsDF['Labels réels'],
                                      LabelsDF['Labels KMeans'])

            Scores.loc[row, 'Algo'] = a
            Scores.loc[row, 'perplexityTSNE'] = str(p)
            Scores.loc[row, 'ARI'] = ARI
            row += 1

            kmeansfig = px.scatter(tsneLab,
                                   x=0,
                                   y=1,
                                   title='KMeans t-SNE{} {}'.format(p, a),
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
                kmeansfig.write_image('./Figures/kmean{}{}.pdf'.format(p, a))

            print('ARI :{}'.format(ARI))

    return Scores


# %%
Scores = clustering(['SIFT', 'ORB', 'VGG', 'XEPT', 'IV3'])
# %%
fig = px.bar(
    Scores,
    x='perplexityTSNE',
    y='ARI',
    color='Algo',
    barmode='group',
    title=
    "Comparaison des scores en fonction<br>de l'algorithme et de la perplexité"
)
fig.show(renderer='notebook')
if write_data is True:
    fig.write_image('./Figures/CompareIMGScores.pdf')

# %%
