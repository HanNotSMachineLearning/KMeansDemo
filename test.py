# Importeer de modulen
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from sklearn import datasets

iris = datasets.load_iris()
x = scale(iris.data)

# Maak het model
clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(x)

# Plot de uitkomst van het model
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length',
                   'Sepal_Width', 'Petal_Length', 'Petal_Width']
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

# Maak het Ground truth classification plot
plt.subplot(1, 2, 1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width,
            c=color_theme[iris.target], s=50)

plt.title('Ground truth classification')

# Maak het K-Means plot
plt.subplot(1, 2, 2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width,
            c=color_theme[clustering.labels_], s=50)
plt.title('K-Means classification')

plt.show()
