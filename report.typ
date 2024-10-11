#block[
= Week 4: Interpretable Machine Learning for Data Science
<week-4-interpretable-machine-learning-for-data-science>
#strong[Problem];: You have been mandated by a large wine-making company
in Valais to discover the key chemical factors that determine the
quality of wine and build an interpretable model that will help their
cellar masters make decisions daily.

== Settings things up (15\')
<settings-things-up-15>
This week will require quite a lot of autonomy on your part, but we will
guide you with this high-level notebook. First, take the following
steps:

- Install #link("https://python-poetry.org")[Poetry];.

- Then use Poetry to create a virtual environment:

  ```sh
  poetry install
  ```

- Then restart VS Code and add the kernel that corresponds to the
  environment created by Poetry.

]
#block[
Then, let\'s set up #link("https://github.com/psf/black")[black];, which
is a highly encouraged best-practice for all your Python projects. That
way, you never have to worry and debate about code formatting anymore.
By using it, you agree to cede control over minutiae of hand-formatting.
In return, Black gives you speed, determinism, and freedom from
`pycodestyle` nagging about formatting. You will save time and mental
energy for more important matters.

]
#block[
```python
import jupyter_black

jupyter_black.load()
```

]
#block[
Here are the libraries you will most likely need and use during this
week:

- `numpy` for basic scientific computing and `scipy` for statistical
  testing.
- `pandas` or `polars` for dataset manipulation. Polars is highly
  recommended, because it is
  #link("https://github.com/ddotta/awesome-polars")[awesome];.
  Instructions below will refer to the Polars API.
- `seaborn` for statistical data visualization, but `matplotlib` is
  always needed anyway. Use both!
- `shap` will be used for
  #link("https://shap.readthedocs.io/en/stable/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html")[interpretability];.
- `sklearn` and `xgboost` will be used for training models. You may
  import them later when you need them.

]
#block[
```python
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
```

]
#block[
== Fetch the data (15\')
<fetch-the-data-15>
Here we have a very nice package that can do everything for us (aka
`ucimlrepo`). Let\'s use it!

Take a look at
#link("https://archive.ics.uci.edu/dataset/186/wine+quality")[the website]
for details.

]
#block[
```python
from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
y = wine_quality.data.targets
X = wine_quality.data.original  # color is not in `features`?
X.drop(columns=["quality"], inplace=True)
```

]
#block[
```python
# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)
```

#block[
```
{'uci_id': 186, 'name': 'Wine Quality', 'repository_url': 'https://archive.ics.uci.edu/dataset/186/wine+quality', 'data_url': 'https://archive.ics.uci.edu/static/public/186/data.csv', 'abstract': 'Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009], http://www3.dsi.uminho.pt/pcortez/wine/).', 'area': 'Business', 'tasks': ['Classification', 'Regression'], 'characteristics': ['Multivariate'], 'num_instances': 4898, 'num_features': 11, 'feature_types': ['Real'], 'demographics': [], 'target_col': ['quality'], 'index_col': None, 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 2009, 'last_updated': 'Wed Nov 15 2023', 'dataset_doi': '10.24432/C56S3T', 'creators': ['Paulo Cortez', 'A. Cerdeira', 'F. Almeida', 'T. Matos', 'J. Reis'], 'intro_paper': {'ID': 252, 'type': 'NATIVE', 'title': 'Modeling wine preferences by data mining from physicochemical properties', 'authors': 'P. Cortez, A. Cerdeira, Fernando Almeida, Telmo Matos, J. Reis', 'venue': 'Decision Support Systems', 'year': 2009, 'journal': None, 'DOI': None, 'URL': 'https://www.semanticscholar.org/paper/Modeling-wine-preferences-by-data-mining-from-Cortez-Cerdeira/bf15a0ccc14ac1deb5cea570c870389c16be019c', 'sha': None, 'corpus': None, 'arxiv': None, 'mag': None, 'acl': None, 'pmid': None, 'pmcid': None}, 'additional_info': {'summary': 'The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).\n\nThese datasets can be viewed as classification or regression tasks.  The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.\n', 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': 'For more information, read [Cortez et al., 2009].\r\nInput variables (based on physicochemical tests):\r\n   1 - fixed acidity\r\n   2 - volatile acidity\r\n   3 - citric acid\r\n   4 - residual sugar\r\n   5 - chlorides\r\n   6 - free sulfur dioxide\r\n   7 - total sulfur dioxide\r\n   8 - density\r\n   9 - pH\r\n   10 - sulphates\r\n   11 - alcohol\r\nOutput variable (based on sensory data): \r\n   12 - quality (score between 0 and 10)', 'citation': None}}
                    name     role         type demographic  \
0          fixed_acidity  Feature   Continuous        None   
1       volatile_acidity  Feature   Continuous        None   
2            citric_acid  Feature   Continuous        None   
3         residual_sugar  Feature   Continuous        None   
4              chlorides  Feature   Continuous        None   
5    free_sulfur_dioxide  Feature   Continuous        None   
6   total_sulfur_dioxide  Feature   Continuous        None   
7                density  Feature   Continuous        None   
8                     pH  Feature   Continuous        None   
9              sulphates  Feature   Continuous        None   
10               alcohol  Feature   Continuous        None   
11               quality   Target      Integer        None   
12                 color    Other  Categorical        None   

               description units missing_values  
0                     None  None             no  
1                     None  None             no  
2                     None  None             no  
3                     None  None             no  
4                     None  None             no  
5                     None  None             no  
6                     None  None             no  
7                     None  None             no  
8                     None  None             no  
9                     None  None             no  
10                    None  None             no  
11  score between 0 and 10  None             no  
12            red or white  None             no  
```

]
]
#block[
Now, let\'s check that the data have the correct shape to ensure they
have been loaded as expected.

Calculate how many samples and features we have in total, how many are
red or white wines, how many are good or bad wines, etc.

]
#block[
```python
# How many samples
print("Samples count: ", X.shape[1])

# How many features
print("Features count:", y.shape[1])

# How many red/white wines
print("Red/White wines count:", X["color"].value_counts())

# How many quality levels
print("Quality levels count:", y["quality"].value_counts())
```

#block[
```
Samples count:  12
Features count: 1
Red/White wines count: color
white    4898
red      1599
Name: count, dtype: int64
Quality levels count: quality
6    2836
5    2138
7    1079
4     216
8     193
3      30
9       5
Name: count, dtype: int64
```

]
]
#block[
```python
sns.histplot(y["quality"])
```

#block[
```
<Axes: xlabel='quality', ylabel='Count'>
```

]
]
#block[
- 4898 white wines and 1599 red wines.

- Very few high and low quality wines, most are average.

\--\> The dataset is imbalanced.

]
#block[
== Data Exploration (1h30)
<data-exploration-1h30>
We now will inspect the features one-by-one, and try to understand their
dynamics, especially between white and red wines.

- Use `Dataframe.describe` to display statistics on each feature. Do the
  same for red wines only, and white wines only. Do you notice any clear
  difference?
- Compute the effect size by computing the
  #link("https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference")[strictly standardized mean difference]
  (SSMD) between the red and white wines for each feature.

]
#block[
```python
X.describe()
```

#block[
```
       fixed_acidity  volatile_acidity  citric_acid  residual_sugar  \
count    6497.000000       6497.000000  6497.000000     6497.000000   
mean        7.215307          0.339666     0.318633        5.443235   
std         1.296434          0.164636     0.145318        4.757804   
min         3.800000          0.080000     0.000000        0.600000   
25%         6.400000          0.230000     0.250000        1.800000   
50%         7.000000          0.290000     0.310000        3.000000   
75%         7.700000          0.400000     0.390000        8.100000   
max        15.900000          1.580000     1.660000       65.800000   

         chlorides  free_sulfur_dioxide  total_sulfur_dioxide      density  \
count  6497.000000          6497.000000           6497.000000  6497.000000   
mean      0.056034            30.525319            115.744574     0.994697   
std       0.035034            17.749400             56.521855     0.002999   
min       0.009000             1.000000              6.000000     0.987110   
25%       0.038000            17.000000             77.000000     0.992340   
50%       0.047000            29.000000            118.000000     0.994890   
75%       0.065000            41.000000            156.000000     0.996990   
max       0.611000           289.000000            440.000000     1.038980   

                pH    sulphates      alcohol  
count  6497.000000  6497.000000  6497.000000  
mean      3.218501     0.531268    10.491801  
std       0.160787     0.148806     1.192712  
min       2.720000     0.220000     8.000000  
25%       3.110000     0.430000     9.500000  
50%       3.210000     0.510000    10.300000  
75%       3.320000     0.600000    11.300000  
max       4.010000     2.000000    14.900000  
```

]
]
#block[
```python
X[X["color"] == "white"].describe()
```

#block[
```
       fixed_acidity  volatile_acidity  citric_acid  residual_sugar  \
count    4898.000000       4898.000000  4898.000000     4898.000000   
mean        6.854788          0.278241     0.334192        6.391415   
std         0.843868          0.100795     0.121020        5.072058   
min         3.800000          0.080000     0.000000        0.600000   
25%         6.300000          0.210000     0.270000        1.700000   
50%         6.800000          0.260000     0.320000        5.200000   
75%         7.300000          0.320000     0.390000        9.900000   
max        14.200000          1.100000     1.660000       65.800000   

         chlorides  free_sulfur_dioxide  total_sulfur_dioxide      density  \
count  4898.000000          4898.000000           4898.000000  4898.000000   
mean      0.045772            35.308085            138.360657     0.994027   
std       0.021848            17.007137             42.498065     0.002991   
min       0.009000             2.000000              9.000000     0.987110   
25%       0.036000            23.000000            108.000000     0.991723   
50%       0.043000            34.000000            134.000000     0.993740   
75%       0.050000            46.000000            167.000000     0.996100   
max       0.346000           289.000000            440.000000     1.038980   

                pH    sulphates      alcohol  
count  4898.000000  4898.000000  4898.000000  
mean      3.188267     0.489847    10.514267  
std       0.151001     0.114126     1.230621  
min       2.720000     0.220000     8.000000  
25%       3.090000     0.410000     9.500000  
50%       3.180000     0.470000    10.400000  
75%       3.280000     0.550000    11.400000  
max       3.820000     1.080000    14.200000  
```

]
]
#block[
```python
X[X["color"] == "red"].describe()
```

#block[
```
       fixed_acidity  volatile_acidity  citric_acid  residual_sugar  \
count    1599.000000       1599.000000  1599.000000     1599.000000   
mean        8.319637          0.527821     0.270976        2.538806   
std         1.741096          0.179060     0.194801        1.409928   
min         4.600000          0.120000     0.000000        0.900000   
25%         7.100000          0.390000     0.090000        1.900000   
50%         7.900000          0.520000     0.260000        2.200000   
75%         9.200000          0.640000     0.420000        2.600000   
max        15.900000          1.580000     1.000000       15.500000   

         chlorides  free_sulfur_dioxide  total_sulfur_dioxide      density  \
count  1599.000000          1599.000000           1599.000000  1599.000000   
mean      0.087467            15.874922             46.467792     0.996747   
std       0.047065            10.460157             32.895324     0.001887   
min       0.012000             1.000000              6.000000     0.990070   
25%       0.070000             7.000000             22.000000     0.995600   
50%       0.079000            14.000000             38.000000     0.996750   
75%       0.090000            21.000000             62.000000     0.997835   
max       0.611000            72.000000            289.000000     1.003690   

                pH    sulphates      alcohol  
count  1599.000000  1599.000000  1599.000000  
mean      3.311113     0.658149    10.422983  
std       0.154386     0.169507     1.065668  
min       2.740000     0.330000     8.400000  
25%       3.210000     0.550000     9.500000  
50%       3.310000     0.620000    10.200000  
75%       3.400000     0.730000    11.100000  
max       4.010000     2.000000    14.900000  
```

]
]
#block[
- We can notice that there\'s a big difference in the mean sulfure
  dioxide content (both free sulfur and total sulfure) between red and
  white wines.
- Some features are very similar, for example alcohol content.
- Some features also have very extreme outliers, for example residual
  sugar (for both red and white wines).

]
#block[
```python
def ssmd(a, b, feature):
    a = a[feature]
    b = b[feature]
    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std()
    b_std = b.std()

    return (a_mean - b_mean) / np.sqrt(a_std**2 + b_std**2)


ssmd_values = {}
for feature in X.columns:
    if feature == "color":
        continue

    res = ssmd(X[X["color"] == "white"], X[X["color"] == "red"], feature)
    ssmd_values[feature] = res
    print(f"SSMD for {feature}: {res}")
```

#block[
```
SSMD for fixed_acidity: -0.7570984913882014
SSMD for volatile_acidity: -1.2146180859422455
SSMD for citric_acid: 0.2756520281549347
SSMD for residual_sugar: 0.7318262377213726
SSMD for chlorides: -0.803525280055768
SSMD for free_sulfur_dioxide: 0.9732927083649252
SSMD for total_sulfur_dioxide: 1.709893545158607
SSMD for density: -0.7689026646492044
SSMD for pH: -0.5688537900809766
SSMD for sulphates: -0.8236123900161837
SSMD for alcohol: 0.056074487578264144
```

]
]
#block[
- Higher magnitude --\> feature is more important to distinguish between
  red and white wines.

- Towards negative --\> mean is higher for red wines.

- Towards positive --\> mean is higher for white wines.

- We can notice the same difference between features for red and white
  wines that we did earlier using the describe method.

]
#block[
Now let\'s go a bit deeper into the same analysis, using more visual
tools:

- For every feature, plot boxplots, violinplots or histograms for red
  and white wines. What can you infer? #strong[If you feel a bit more
  adventurous];, plot the Cumulative Distribution Function (CDF) of the
  feature for white and red wines, and compute the
  #link("https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html")[Kullback-Leibler divergence]
  (or entropy) between them. Explain why this might be useful.
- Plot the correlation matrix of all features as heatmaps, one for red
  and one for white wines. How do they differ? What can you infer?

]
#block[
```python
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # just to ignore some deprecation warnings

numerical_features = X.columns.drop("color")

g = sns.FacetGrid(
    X.melt(id_vars="color", value_vars=numerical_features),
    col="variable",
    col_wrap=4,
    height=3,
    sharey=False,
)

palette = {"red": "red", "white": "white"}
g.map(sns.violinplot, "color", "value", palette=palette)

g.set_titles("{col_name}")
g.set_axis_labels("Wine Color", "Value")

# red/white x-axis labels don't show for all plots otherwise?
for ax in g.axes.flat:
    ax.tick_params(labelbottom=True)

warnings.filterwarnings("default")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/e6bfdc995ad2ffe4ccda308600391c359dcb3f57.png"))

]
#block[
#box(image("--template=isc_templace.typ/2a56aa3d7c18320a1508c573da9a234a84a43495.png"))

]
]
#block[
```python
# plot the correlation matrix of the features
plt.figure(figsize=(10, 8))
sns.heatmap(X[numerical_features].corr(), annot=True, cmap="coolwarm")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/a5b35975ade40e2864b7853a2c3b77586812c7af.png"))

]
]
#block[
== Data Exploration using Unsupervised Learning (3h)
<data-exploration-using-unsupervised-learning-3h>
We first explore the data in an unsupervised fashion. Start by creating
a heatmap of the average feature value for red and white wines. Can you
spot an easy way to differentiate between reds and whites?

]
#block[
```python
X_red = X[X["color"] == "red"]
X_white = X[X["color"] == "white"]

avg_red = X_red[numerical_features].mean()
avg_white = X_white[numerical_features].mean()
avg_diff = avg_red - avg_white

plt.figure(figsize=(10, 1))
sns.heatmap([avg_red, avg_white], annot=True, cmap="vlag", cbar=False, fmt=".2f")
plt.title("Difference between red and white wines")
plt.xticks(
    rotation=45, labels=numerical_features, ticks=np.arange(len(numerical_features))
)
plt.yticks(ticks=[0.5, 1.5], labels=["Red", "White"], rotation=0)
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/2beeb7b221fa64654e5bb56c9e99bfa33a7b8fe5.png"))

]
]
#block[
?

]
#block[
=== Using PCA to reduce the dimensionality
<using-pca-to-reduce-the-dimensionality>
Use PCA to reduce the dimensionality of data. Do not forget that it
requires data normalization (centering on the mean and scaling to unit
variance). Plot the whole dataset onto the two principal components and
color it by wine color. What does it tell you?

Project the unit vectors that correspond to each vector onto the
principal components, using the same transformation. What does it tell
you about the relative feature importance? Does it match the
observations you made previously?

]
#block[
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

palette = {"red": "red", "white": "yellow"}
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], palette=palette, hue=X["color"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of wine dataset")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/a325e0e3c4c9c2126657a99b115cf0c2eac2ac31.png"))

]
]
#block[
```python
from adjustText import adjust_text


unit_vectors = np.eye(X_scaled.shape[1])
unit_vectors_pca = pca.transform(unit_vectors)

loadings = pca.components_.T

# Plot the feature vectors on the same PCA plot
palette = {"red": "orange", "white": "yellow"}
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], palette=palette, hue=X["color"], alpha=0.33
)

texts = []
for i, feature in enumerate(numerical_features):
    plt.arrow(
        0,
        0,
        loadings[i, 0] * 10,
        loadings[i, 1] * 10,
        color="gray",
        alpha=0.33,
        head_width=0.3,
    )
    texts.append(
        plt.text(loadings[i, 0] * 10, loadings[i, 1] * 10, feature, fontsize=11)
    )

adjust_text(texts)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of wine dataset with feature vectors")
plt.text(
    1,
    13,
    "(Feature vectors amplified 10-fold for readability)",
    fontsize=11,
    horizontalalignment="center",
    color="black",
    alpha=0.7,
)
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/5bbed8679373f6401201587fe30973bb97f63ce2.png"))

]
]
#block[
- We can see that both sulfure dioxide features tend towards where the
  white wines are, while the fixed and volatile acidity and other
  features tend towards where the red wines are.
- This is consistent with the SSMD analysis we did earlier.

]
#block[
=== Cluster the data in 2-dimensional space
<cluster-the-data-in-2-dimensional-space>
Use k-means to cluster the data into 2 clusters and plot the same view
as before, but with a coloring that corresponds to the cluster
memberships.

Assuming that the cluster assignments are predictions of a model, what
is the performance you can achieve in terms of mutual information score,
accuracy, and f1 score?

]
#block[
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
X_clustered = kmeans.predict(X_scaled)

palette = {0: "green", 1: "blue"}
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], palette=palette, hue=X_clustered)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans clustering of wine dataset")
plt.legend(labels=["Cluster 0", "Cluster 1"])
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/514125360a688e2c21ec4e71cf9fbf0a26f435d3.png"))

]
]
#block[
```python
X["color_encoded"] = X["color"].map({"red": 0, "white": 1})
```

]
#block[
```python
# transform the cluster centers to PCA space
kmeans_centers_pca_space = pca.transform(kmeans.cluster_centers_)

# transform the red and white wines points to PCA space
pca_white_points = pca.transform(X_scaled[X["color_encoded"] == 1])
pca_red_points = pca.transform(X_scaled[X["color_encoded"] == 0])

# calculate the center of the red and white wines in PCA space
pca_white_center = pca_white_points.mean(axis=0)
pca_red_center = pca_red_points.mean(axis=0)

# find out which cluster is closer to the red and white wines
distances_to_red = np.linalg.norm(kmeans_centers_pca_space - pca_red_center, axis=1)
distances_to_white = np.linalg.norm(kmeans_centers_pca_space - pca_white_center, axis=1)

cluster_red = np.argmin(distances_to_red)
cluster_white = np.argmin(distances_to_white)

print(f"Cluster {cluster_red} is closer to red wines")
print(f"Cluster {cluster_white} is closer to white wines")
```

#block[
```
Cluster 0 is closer to red wines
Cluster 1 is closer to white wines
```

]
]
#block[
```python
from sklearn.metrics import classification_report
from sklearn.metrics import mutual_info_score

mis = mutual_info_score(X["color_encoded"], X_clustered)

# if the cluster that is closer to red wines is cluster 1, we need to invert the cluster labels
# (because we want 0 to represent red wines)
if cluster_red == 1:
    X_clustered = 1 - X_clustered

print(
    classification_report(
        X["color_encoded"], X_clustered, target_names=["red", "white"]
    )
)
print(f"Mutual information score: {mis}")
```

#block[
```
              precision    recall  f1-score   support

         red       0.96      0.98      0.97      1599
       white       1.00      0.99      0.99      4898

    accuracy                           0.99      6497
   macro avg       0.98      0.99      0.98      6497
weighted avg       0.99      0.99      0.99      6497

Mutual information score: 0.4911471062925098
```

]
]
#block[
Now, we are going to train a #strong[supervised] linear classification
model using `sklearn`, and compare the results with the approach using
clustering.

- Set up a train/test dataset using
  `sklearn.model_selection.train_test_split`.
- Use `GridSearchCV` to perform a cross-validation of the model\'s
  regularization `C`.
- Compare the test and train performance at the end. Does the model
  suffer from any overfitting?
- Analyze the test performance specifically. What can you conclude about
  this general problem of recognizing white vs red wines?

]
#block[
```python
# the model's not balanced on the label (color), so we need to balance it
X_white = X[X["color"] == "white"]
X_red = X[X["color"] == "red"]
white_count = X_white.shape[0]
red_count = X_red.shape[0]

if white_count > red_count:
    X_white_downsampled = X_white.sample(n=red_count, random_state=42)
    X_downsampled = pd.concat([X_red, X_white_downsampled])
else:
    X_red_downsampled = X_red.sample(n=white_count, random_state=42)
    X_downsampled = pd.concat([X_red_downsampled, X_white])

X_downsampled.drop(columns=["color"], inplace=True)
```

]
#block[
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    grid = {
        "C": np.linspace(0.1, 2, 100),
    }

    logreg = linear_model.LogisticRegression(solver="liblinear")
    logreg_cv = GridSearchCV(logreg, grid, cv=5)
    logreg_cv.fit(X_train, y_train)

    return logreg_cv, X_train, X_test, y_train, y_test


logreg_cv, X_train, X_test, y_train, y_test = train(
    X_downsampled, X_downsampled["color_encoded"]
)

print("Hyperparameters: ", logreg_cv.best_params_)
print("Best score: ", logreg_cv.best_score_)
print("Test score: ", logreg_cv.score(X_test, y_test))
```

#block[
```
Hyperparameters:  {'C': 0.5414141414141413}
Best score:  0.9994791666666668
Test score:  0.9984375
```

]
]
#block[
```python
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.2)
train_sizes, train_scores, test_scores = learning_curve(
    logreg_cv.best_estimator_, X_train, y_train, cv=cv
)

display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="accuracy",
)
display.plot()
plt.show()
```

#block[
```
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
c:\Users\Dimitri\AppData\Local\pypoetry\Cache\virtualenvs\301-explainable-ml-MfwWe0RQ-py3.12\Lib\site-packages\sklearn\utils\validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
```

]
#block[
#box(image("--template=isc_templace.typ/2bf532f505251797003fef72192e5be0d7229bc6.png"))

]
]
#block[
=== Basic model interpretability: inspecting the model
<basic-model-interpretability-inspecting-the-model>
As a first step towards intepretability of the model predictions, let\'s
take a look at the coefficients of the model. What is the most important
feature from this perspective? How do you interpret positive or negative
coefficients?

Is it compatible with what you have seen so far? Do you have an
explanation why that might be?

]
#block[
```python
print("Model coefficients: ")
coeff_values = list(zip(numerical_features, logreg_cv.best_estimator_.coef_[0]))
coeff_values
```

#block[
```
Model coefficients: 
```

]
#block[
```
[('fixed_acidity', -0.4812151499783624),
 ('volatile_acidity', -1.3014773631281895),
 ('citric_acid', 0.32084778255316787),
 ('residual_sugar', 0.08103276225408546),
 ('chlorides', -0.21840446417463832),
 ('free_sulfur_dioxide', 0.006769638000773905),
 ('total_sulfur_dioxide', 0.024844086519965145),
 ('density', -0.007330019695941643),
 ('pH', -0.8378170045553838),
 ('sulphates', -0.91576796841402),
 ('alcohol', 0.12476910007905598)]
```

]
]
#block[
```python
print("SSMD values: ")
for feature, ssmd_value in ssmd_values.items():
    print(f"{feature}: {ssmd_value}")
```

#block[
```
SSMD values: 
fixed_acidity: -0.7570984913882014
volatile_acidity: -1.2146180859422455
citric_acid: 0.2756520281549347
residual_sugar: 0.7318262377213726
chlorides: -0.803525280055768
free_sulfur_dioxide: 0.9732927083649252
total_sulfur_dioxide: 1.709893545158607
density: -0.7689026646492044
pH: -0.5688537900809766
sulphates: -0.8236123900161837
alcohol: 0.056074487578264144
```

]
]
#block[
```python
ssmd_values = pd.Series(ssmd_values)
coeff_values = pd.Series(dict(coeff_values))

fig, ax = plt.subplots(figsize=(10, 5))
ssmd_values.plot(kind="bar", color="blue", ax=ax, alpha=0.5, width=0.25, position=1)
coeff_values.plot(kind="bar", color="red", ax=ax, alpha=0.5, width=0.25, position=0)
plt.title("SSMD values and model coefficients")
plt.xticks(rotation=45, ha="right")
plt.legend(["SSMD", "Coefficient"])
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/30c23fc7783de7739eb0304b5e9734ac33835b4f.png"))

]
#block[
#box(image("--template=isc_templace.typ/d641709092e12096ce87c0a1e83f609ab9c22094.png"))

]
]
#block[
- Features that we determined to be important for red wines (fixed
  acidity, volatile acidity, citric acid, ...) seem to have a similar
  impact on the model\'s prediction as what we determined with the SSMD
  analysis.
- However, features that we determined to be important for white wines
  (free sulfur dioxide, total sulfut dioxide) seem to have no impact on
  the model\'s prediction, which is surprising, and despite the fact
  that they have a high SSMD.

]
#block[
=== Removing features to test their importance
<removing-features-to-test-their-importance>
- What happens if you re-train a model, but remove the most important
  feature in the list?
- What happens if you re-train the model with a `l1` penalty and you use
  more regularization?
- Interpret the results you obtained above from the perspective of the
  business problem. What does it tell you about the key differences
  between a red and white wine?

]
#block[
```python
most_important_features = coeff_values.abs().sort_values(ascending=False)
most_important_features
```

#block[
```
volatile_acidity        1.301477
sulphates               0.915768
pH                      0.837817
fixed_acidity           0.481215
citric_acid             0.320848
chlorides               0.218404
alcohol                 0.124769
residual_sugar          0.081033
total_sulfur_dioxide    0.024844
density                 0.007330
free_sulfur_dioxide     0.006770
dtype: float64
```

]
]
#block[
```python
# remove volatile_acidity and sulphates
X_removed_features = X_downsampled.drop(columns=["volatile_acidity", "sulphates"])

logreg_cv_removed, X_train_removed, X_test_removed, y_train_removed, y_test_removed = (
    train(X_removed_features, X_removed_features["color_encoded"])
)

print("Hyperparameters: ", logreg_cv_removed.best_params_)
print("Best score: ", logreg_cv_removed.best_score_)
print("Test score: ", logreg_cv_removed.score(X_test_removed, y_test_removed))
```

#block[
```
Hyperparameters:  {'C': 0.598989898989899}
Best score:  0.9994791666666668
Test score:  0.99921875
```

]
]
#block[
...

]
#block[
=== Using Shapley values
<using-shapley-values>
Now, use SHAP to explore how the model perceives a \'red\' and \'white\'
wine.

- Use a `beeswarm` plot to analyze the influence of each feature on the
  model\'s output.
- What does the plot tell us about what makes a white wine \'white\' and
  a red wine \'red\'?

]
#block[
```python
explainer = shap.Explainer(
    logreg_cv.best_estimator_.fit(X_train[numerical_features], y_train),
    X_train[numerical_features],
)
shap_values = explainer(X_train[numerical_features])
shap.plots.beeswarm(shap_values, max_display=15)
```

#block[
#box(image("--template=isc_templace.typ/223b9aa40e6c612047d4d7374f2286934bd2989b.png"))

]
]
#block[
The beeswarm graph allows us to see the influence of each feature on the
model\'s output. As observed earlier, the most important feature is the
total sulfur dioxide content; higher values indeed have a big impact on
the model\'s output compared to other features. We can also see that for
other features, more notably the fixed acidity, higher values have a
negative impact on the model\'s output.

]
#block[
- Now use Partial Dependence Plots to see how the expected model output
  varies with the variation of each feature.

]
#block[
```python
for feature in numerical_features:
    shap.partial_dependence_plot(
        feature,
        logreg_cv.best_estimator_.predict,
        X_train[numerical_features],
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
    )
```

#block[
#box(image("--template=isc_templace.typ/9879d4bd312b4f38b3330338d97ec4c5fbff7863.png"))

]
#block[
#box(image("--template=isc_templace.typ/2f87b0f2f55559be72c6bda001066cfed77c2bd8.png"))

]
#block[
#box(image("--template=isc_templace.typ/9270e0552601ab4fa86cddb94164a7b59aa5b6a7.png"))

]
#block[
#box(image("--template=isc_templace.typ/738669a690ae5ee9e5ac5fbfd024cd02eea59fda.png"))

]
#block[
#box(image("--template=isc_templace.typ/31cca2e7f682d32d73aea1a241ca2ac4fd69471b.png"))

]
#block[
#box(image("--template=isc_templace.typ/75fbe8a4b8af22d408116e19d355ebecb3097535.png"))

]
#block[
#box(image("--template=isc_templace.typ/e2b90d20ddfafbbf1296fbb6a84ea7c2aaf447da.png"))

]
#block[
#box(image("--template=isc_templace.typ/dea10b47c282756af0b1e82a0174872a41704dad.png"))

]
#block[
#box(image("--template=isc_templace.typ/edf465fe8f09b5663b9dc47d8824a285cc79b676.png"))

]
#block[
#box(image("--template=isc_templace.typ/9fa438e4ebd8c290be3789bd95c3e53687e7fb12.png"))

]
#block[
#box(image("--template=isc_templace.typ/d28268f0d67f3748b18e54d75a3660b9e770140b.png"))

]
]
#block[
- Now use a waterfall diagram on a specific red and white wine and see
  how the model has made this specific prediction.

]
#block[
```python
predictions = logreg_cv.best_estimator_.predict(X_test[numerical_features])

rightly_classified = X_test[predictions == y_test]

random_red_wine = rightly_classified[rightly_classified["color_encoded"] == 0].sample(1)
random_white_wine = rightly_classified[rightly_classified["color_encoded"] == 1].sample(
    1
)

shap_values_white = explainer(random_white_wine[numerical_features])
shap_values_red = explainer(random_red_wine[numerical_features])

shap.plots.waterfall(shap_values_white[0], show=False)
plt.title("SHAP values for a correctly predicted white wine")
plt.show()

shap.plots.waterfall(shap_values_red[0], show=False)
plt.title("SHAP values for a correctly predicted red wine")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/0657d43551b679291bc375a24b56c77c6a5d25aa.png"))

]
#block[
#box(image("--template=isc_templace.typ/2ddd9a61f29d52e480549d22899192b3a976fb4f.png"))

]
]
#block[
- Now, let\'s take an example where the model has made an incorrect
  prediction, and see how it made this prediction.

]
#block[
```python
misclassified = X_test[predictions != y_test]

wine = misclassified.sample(1)
color = "red" if wine["color_encoded"].values[0] == 0 else "white"

shap.plots.waterfall(
    explainer(misclassified[numerical_features])[0], show=False, max_display=15
)
plt.title(f"SHAP waterfall for a misclassified {color} wine")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/5585a9b5104a219835060467c00254e09f492042.png"))

]
]
#block[
= Good vs Bad classification (3h)
<good-vs-bad-classification-3h>
We are going to work on a binary classification problem, where all wines
with a quality higher than 6 are considered as \"good\" and other are
considered as \"bad\".

- Prepare a dataset with a new column `binary_quality` that corresponds
  to the above definition.

]
#block[
```python
df_binary_quality = y.map(lambda x: 1 if x > 6 else 0)
```

]
#block[
One question that we might ask right away is:

- Is there any correlation of the quality and the color of the wine?

Ideally, there should be almost none. Why could it be a problem
otherwise?

]
#block[
```python
# display a correlation matrix with color and binary

X_derp = X.copy()
X_derp["color_red"] = X_derp["color"].map({"red": 1, "white": 0})
X_derp["color_white"] = X_derp["color"].map({"red": 0, "white": 1})
X_derp["quality_good"] = df_binary_quality["quality"].map({1: 1, 0: 0})
X_derp["quality_bad"] = df_binary_quality["quality"].map({1: 0, 0: 1})

sns.heatmap(
    X_derp[["quality_bad", "quality_good", "color_red", "color_white"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation matrix with color and binary quality")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/fdd66fb8aaa674897ae4420d2935056107d55534.png"))

]
]
#block[
If it turns out that there are significantly more bad red wines than bad
white wines or vice versa, what are the implications for your analysis?

- Plot a heatmap of the mean feature value for bad and good wines, like
  we did before for red and white wines.
- Plot two heatmaps, one for red and white wines. How do they differ?
  What kind of issue can it cause?

]
#block[
```python
X_good = X[df_binary_quality["quality"] == 1]
X_bad = X[df_binary_quality["quality"] == 0]

avg_good = X_good[numerical_features].mean()
avg_bad = X_bad[numerical_features].mean()

plt.figure(figsize=(10, 1))
sns.heatmap([avg_good, avg_bad], annot=True, cmap="vlag", cbar=False)
plt.title("Difference between good and bad wines")
plt.xticks(
    rotation=45, labels=numerical_features, ticks=np.arange(len(numerical_features))
)
plt.yticks(ticks=[0.5, 1.5], labels=["Bad", "Good"], rotation=0)
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/74bc5f481c2df66e865ab3d9404458007b49c4a8.png"))

]
]
#block[
It is a lot more difficult now to tell apart good from bad wines. Let\'s
turn to a more complex model, which is a
#link("https://en.wikipedia.org/wiki/Gradient_boosting")[Gradient Boosting]
#link("https://xgboost.readthedocs.io/en/stable/tutorials/model.html")[Trees];.
For the sake of interpretability, design your notebook so that you can
easily filter on only white and red wines and perform again the entire
procedure.

Let\'s first train a XGBClassifier model to distinguish between good and
bad wines. Make sure to use the same best-practices (train/test split,
cross-validation) as we did before. Note that the regularization of the
GBTs is a lot more complex than for Logistic Regression. Test the
following parameters:

```py
param_grid = {
  "max_depth": [3, 4, 5],  # Focus on shallow trees to reduce complexity
  "learning_rate": [0.01, 0.05, 0.1],  # Slower learning rates
  "n_estimators": [50, 100],  # More trees but keep it reasonable
  "min_child_weight": [1, 3],  # Regularization to control split thresholds
  "subsample": [0.7, 0.9],  # Sampling rate for boosting
  "colsample_bytree": [0.7, 1.0],  # Sampling rate for columns
  "gamma": [0, 0.1],  # Regularization to penalize complex trees
}
```

]
#block[
```python
from sklearn.preprocessing import RobustScaler


param_grid = {
    "max_depth": [3, 4, 5],  # Focus on shallow trees to reduce complexity
    "learning_rate": [0.01, 0.05, 0.1],  # Slower learning rates
    "n_estimators": [50, 100],  # More trees but keep it reasonable
    "min_child_weight": [1, 3],  # Regularization to control split thresholds
    "subsample": [0.7, 0.9],  # Sampling rate for boosting
    "colsample_bytree": [0.7, 1.0],  # Sampling rate for columns
    "gamma": [0, 0.1],  # Regularization to penalize complex trees
}

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X[numerical_features])
X_scaled = pd.DataFrame(X_scaled, columns=X[numerical_features].columns)

xgb_model = xgb.XGBClassifier(objective="binary:logistic")

grid = GridSearchCV(xgb_model, param_grid, cv=10, n_jobs=-1, scoring="f1")

f = X.columns.drop("color")
X_red = X_scaled[X["color"] == "red"]
y_red = df_binary_quality[X["color"] == "red"]

X_train, X_test, y_train, y_test = train_test_split(
    X_red, y_red, test_size=0.2, stratify=y_red
)

grid.fit(X_train, y_train)

print("Best hyperparameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)
print("Test score: ", grid.score(X_test, y_test))
```

#block[
```
Best hyperparameters:  {'colsample_bytree': 0.7, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.7}
Best score:  0.6418544653027412
Test score:  0.56
```

]
]
#block[
- Analyze the results (test and train), validate whether there is
  overfitting.

]
#block[
```python
train_sizes, train_scores, test_scores = learning_curve(
    grid.best_estimator_,
    X_train,
    y_train,
    cv=ShuffleSplit(n_splits=20, test_size=0.2),
    scoring="f1",
)

display = LearningCurveDisplay(
    train_sizes=train_sizes,
    train_scores=train_scores,
    test_scores=test_scores,
    score_name="f1-score",
)
display.plot()
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/163717f4158014aae9be6b12b6481f2984fbc890.png"))

]
]
#block[
OK nice overfitting, very good. That\'s what we like to see.

]
#block[
```python
accuracy_test = sklearn.metrics.accuracy_score(y_test, grid.predict(X_test))
accuracy_train = sklearn.metrics.accuracy_score(y_train, grid.predict(X_train))

print(f"Accuracy on test set: {accuracy_test}")
print(f"Accuracy on train set: {accuracy_train}")

recall_test = sklearn.metrics.recall_score(y_test, grid.predict(X_test))
recall_train = sklearn.metrics.recall_score(y_train, grid.predict(X_train))

print(f"Recall on test set: {recall_test}")
print(f"Recall on train set: {recall_train}")

precision_test = sklearn.metrics.precision_score(y_test, grid.predict(X_test))
precision_train = sklearn.metrics.precision_score(y_train, grid.predict(X_train))

print(f"Precision on test set: {precision_test}")
print(f"Precision on train set: {precision_train}")

f1_test = sklearn.metrics.f1_score(y_test, grid.predict(X_test))
f1_train = sklearn.metrics.f1_score(y_train, grid.predict(X_train))

print(f"F1 score on test set: {f1_test}")
print(f"F1 score on train set: {f1_train}")

print("Classification report on test set:")
print(classification_report(y_test, grid.predict(X_test), target_names=["bad", "good"]))

print("Classification report on train set:")
print(
    classification_report(y_train, grid.predict(X_train), target_names=["bad", "good"])
)
```

#block[
```
Accuracy on test set: 0.896875
Accuracy on train set: 0.9921813917122753
Recall on test set: 0.4883720930232558
Recall on train set: 0.9482758620689655
Precision on test set: 0.65625
Precision on train set: 0.9939759036144579
F1 score on test set: 0.56
F1 score on train set: 0.9705882352941176
Classification report on test set:
              precision    recall  f1-score   support

         bad       0.92      0.96      0.94       277
        good       0.66      0.49      0.56        43

    accuracy                           0.90       320
   macro avg       0.79      0.72      0.75       320
weighted avg       0.89      0.90      0.89       320

Classification report on train set:
              precision    recall  f1-score   support

         bad       0.99      1.00      1.00      1105
        good       0.99      0.95      0.97       174

    accuracy                           0.99      1279
   macro avg       0.99      0.97      0.98      1279
weighted avg       0.99      0.99      0.99      1279

```

]
]
#block[
== Interpretability with SHAP (2h)
<interpretability-with-shap-2h>
- Plot the feature importance (gain and cover) from the XGBoost model.
  What can you conclude?

]
#block[
```python
xgb.plot_importance(grid.best_estimator_, importance_type="cover")
plt.title("Feature importance by cover")
plt.show()
xgb.plot_importance(grid.best_estimator_, importance_type="gain")
plt.title("Feature importance by gain")
plt.show()
```

#block[
#box(image("--template=isc_templace.typ/cd20e912fc8d002ddcdccead610fbee0b9f53e46.png"))

]
#block[
#box(image("--template=isc_templace.typ/4da8ded076600283f64c25c6edbef4d6dd24a875.png"))

]
]
#block[
==== TODO
<todo>
]
#block[
- Use SHAP\'s `TreeExplainer` to compute feature importance (Shapley
  values). Do you see any difference with XGBoost\'s feature
  importances?
- Produce different plots to analyze Shapley values:
  - A bar plot that summarizes the mean absolute value of each feature.
  - A beeswarm plot that shows the shapley value for every sample and
    every feature.
  - A
    #link("https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/heatmap.html#heatmap-plot")[heatmap plot]
    that indicates how different feature patterns influence the model\'s
    output.
- Based on the above results, what makes a wine \'good\' or \'bad\'?

]
#block[
```python
explainer = shap.TreeExplainer(grid.best_estimator_)
shap_values = explainer(X_train)
```

]
#block[
```python
shap.summary_plot(shap_values, X_train, plot_type="bar")
```

#block[
#box(image("--template=isc_templace.typ/04dd905604a9b0984f6f60e769110ad6d20a37cc.png"))

]
]
#block[
```python
shap.summary_plot(shap_values, X_train[numerical_features])
```

#block[
#box(image("--template=isc_templace.typ/83d0ce8f1f8e1f7a46d8dd2291fa8b4329e740c9.png"))

]
]
#block[
```python
# shap.heatmap_plot(shap_values)
random_sample = X_train[numerical_features].sample(50)
shap_values_subset = explainer(random_sample)
shap.plots.heatmap(shap_values_subset, max_display=15)
```

#block[
#box(image("--template=isc_templace.typ/5aabdb9536da1fc17663227306e4fe67daed2f00.png"))

]
#block[
```
<Axes: xlabel='Instances'>
```

]
]
#block[
- Now use Partial Dependence Plots to see how the expected model output
  varies with the variation of each feature.
- How does that modify your perspective on what makes a good or bad
  wine?

]
#block[
```python
for feature in numerical_features:
    shap.partial_dependence_plot(
        feature,
        logreg_cv.best_estimator_.predict,
        X_train[numerical_features],
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
    )
```

#block[
#box(image("--template=isc_templace.typ/c2304e3d663e327ea6ea2550044ec2ac307f66ff.png"))

]
#block[
#box(image("--template=isc_templace.typ/003d894591d41792f5e01a91bab6e8d98269d1dc.png"))

]
#block[
#box(image("--template=isc_templace.typ/a23c775ef8d17641a198bf4aee4eec4728c32769.png"))

]
#block[
#box(image("--template=isc_templace.typ/396731ead9131ce6bf4dd9f20a6f43f2e039c70c.png"))

]
#block[
#box(image("--template=isc_templace.typ/aa10b8d15c43dabf9a375a7d9ce4fc4b57a45e34.png"))

]
#block[
#box(image("--template=isc_templace.typ/0e018fb9e69378cb746c691b348b0f9552d9c063.png"))

]
#block[
#box(image("--template=isc_templace.typ/df45e6fdd3424f364964d8be2b9de8299196d76e.png"))

]
#block[
#box(image("--template=isc_templace.typ/45aadfff57f51483793cb24b5951a04ca0dc6101.png"))

]
#block[
#box(image("--template=isc_templace.typ/ca1caa3bff639b4b522996874dd70b69f27eab80.png"))

]
#block[
#box(image("--template=isc_templace.typ/c37050fad7b69dd6af5ab7793d8218fe889cf3e7.png"))

]
#block[
#box(image("--template=isc_templace.typ/479b46f3f50c4045f69e8bcbb0a2ac4969219d07.png"))

]
]
#block[
- Search for literature or resources that provide indications of the
  chemical structure of good or poor wines. Do your findings match these
  resources?

]
#block[
]
#block[
=== Analyze a few bad wines, and try to see how to make them better
<analyze-a-few-bad-wines-and-try-to-see-how-to-make-them-better>
Pick some of the worst wines, and try to see what make them so bad.
Check out
#link("https://shap.readthedocs.io/en/stable/example_notebooks/api_examples/plots/heatmap.html#heatmap-plot")[`shap.plots.heatmap`]
for some visual tool to do this.

How would you go about improving them?

]
#block[
```python
worst_wines = X_red[df_binary_quality["quality"] == 0].sample(15)

shap_values_worst = explainer(worst_wines[numerical_features])
shap.plots.heatmap(shap_values_worst, max_display=15, show=False)
plt.title("SHAP values for 15 of the worst red wines")
plt.show()
```

#block[
```
C:\Users\Dimitri\AppData\Local\Temp\ipykernel_3936\1902180917.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
  worst_wines = X_red[df_binary_quality["quality"] == 0].sample(15)
```

]
#block[
#box(image("--template=isc_templace.typ/f2736c12325c2e9a4c2ce3ac6a3454116192f7fb.png"))

]
]
#block[
```python
best_wines = X_red[df_binary_quality["quality"] == 1].sample(50)

shap_values_best = explainer(best_wines[numerical_features])
shap.plots.heatmap(shap_values_best, max_display=15, show=False)
plt.title("SHAP values for 15 of the best red wines")
plt.show()
```

#block[
```
C:\Users\Dimitri\AppData\Local\Temp\ipykernel_3936\1428719093.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
  best_wines = X_red[df_binary_quality["quality"] == 1].sample(50)
```

]
#block[
#box(image("--template=isc_templace.typ/82b371741ca994e69554aff84f7bbdba70045c25.png"))

]
]
#block[
= Wrap-up and conclusion
<wrap-up-and-conclusion>
As wrap-up, explain what are your key findings, and make 3
recommendations to the wine maker on how to improve the wines for next
year. How confident are you that making these changes will lead to
better wines? Explain in simple terms to the winemaker the limitations
of your approach in terms of capturing causality.

]
#block[
- Key findings:
  - It\'s much easier to distinguish between red and white wines than
    between good and bad wines.

Most best red wines are characterized by a high alcool content, high
residual sugar content and high free sulfur dioxide.

]
