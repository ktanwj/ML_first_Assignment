from utils import *
from kneed import KneeLocator

# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

df = pd.read_csv('data/bank/train_data.csv')
df = df.drop(columns=['Exited', 'id','CustomerId', 'Surname'])
df.columns.get_loc('Age')
df.columns.get_loc('NumOfProducts')

age = X_train[:,df.columns.get_loc('Age')]
num_p = X_train[:,df.columns.get_loc('NumOfProducts')]
lab = y_train

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(age, num_p, c=y_train.values.flatten(), cmap='bwr', s=50, alpha=0.7, edgecolors='k')
plt.title('Scatter plot of Bank raw data')
plt.colorbar(label='Churn')
plt.xlabel('Age')
plt.ylabel('Number of products')


# weather
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

df = pd.read_csv('data/weather/processed_weather.csv')
features=['month', 'Rainfall', 'Rainfall_prev', 'Evaporation', 'Evaporation_prev', 
          'Sunshine', 'Sunshine_prev', 'WindGustSpeed', 'WindGustSpeed_prev',
           'Humidity9am', 'Humidity9am_prev', 'Cloud9am', 'Cloud9am_prev', 'AverageTemp', 'AverageTemp_prev',
           'WindGustDir_N', 'WindGustDir_S', 'WindGustDir_E', 'WindGustDir_W',
          'Location_Changi', 'Location_Sentosa', 'Location_Tuas', 'Location_Woodlands',
          'Pressure9am_HIGH', 'Pressure9am_LOW', 'Pressure9am_MED']

y = df["RainTomorrow"]
X = df[features]

age = X_train[:,X.columns.get_loc('Humidity9am')]
num_p = X_train[:,X.columns.get_loc('Cloud9am')]
lab = y_train

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(np.log(age), np.log(num_p), c=y_train.values.flatten(), cmap='bwr', s=50, alpha=0.7, edgecolors='k')
plt.title('Scatter plot of Weather raw data')
plt.colorbar(label='RainTomorrow')
plt.xlabel('Humidity')
plt.ylabel('Cloud density')
plt.savefig('src/weather_output/weather_scatter_raw.png')


train_silhouette_scores_kmeans = []
test_silhouette_scores_kmeans = []
k_range = range(3, X_train.shape[1])

for i in k_range:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_train)
    train_pred_labels = kmeans.labels_
    test_pred_labels = kmeans.predict(X_test)

    silhouette_train = silhouette_score(X_train, train_pred_labels)
    silhouette_test = silhouette_score(X_test, test_pred_labels)

    train_silhouette_scores_kmeans.append(silhouette_train)
    test_silhouette_scores_kmeans.append(silhouette_test)

knee_train = KneeLocator(range(3, X_train.shape[1]), train_silhouette_scores_kmeans, curve='convex', direction='decreasing')
kmeans = KMeans(n_clusters=knee_train.knee)
clusters = kmeans.fit_predict(X_train)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(age, num_p, c=y_train.values.flatten(), cmap='bwr', s=50, alpha=0.7, edgecolors='k')
plt.title('Scatter plot of Bank raw data')
plt.colorbar(label='Churn')
plt.xlabel('Age')
plt.ylabel('Number of products')

plt.subplot(1, 2, 2)
plt.scatter(age, num_p, c=clusters, cmap='bwr', s=50, alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, df.columns.get_loc('Age')], kmeans.cluster_centers_[:, df.columns.get_loc('NumOfProducts')], s=300, c='red', marker='X', label='Centroids')
plt.title('Scatter plot of Bank cluster data')
plt.xlabel('Age')
plt.ylabel('Number of products')
plt.legend()