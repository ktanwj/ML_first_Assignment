from utils import *
from kneed import KneeLocator

best_nn_weather_param = {
    'activation': 'tanh',
    'hidden_layer_sizes': (100, 50, 25),
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.01}

best_nn_bank_param = {
    'activation': 'tanh',
    'hidden_layer_sizes': (100, 50, 25),
    'learning_rate': 'constant',
    'learning_rate_init': 0.05}

"""
ICA
"""
# bank tuning (Kmeans)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# Kmeans
dim_red = 'KMEANS'
# Run k-means clustering on the reduced data
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

X_train = kmeans.fit_transform(X_train)
X_test = kmeans.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)

# weather tuning (Kmeans)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

# Kmeans
dim_red = 'KMEANS'
# Run k-means clustering on the reduced data
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

X_train = kmeans.fit_transform(X_train)
X_test = kmeans.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)

# bank (EM)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# EM
dim_red = 'EM'
# Run EM clustering
train_silhouette_scores_em = []
test_silhouette_scores_em = []
k_range = range(3, X_train.shape[1])

for i in k_range:
    em = GaussianMixture(n_components=i)
    em.fit(X_train)
    train_pred_labels = em.predict(X_train)
    test_pred_labels = em.predict(X_test)

    silhouette_train = silhouette_score(X_train, train_pred_labels)
    silhouette_test = silhouette_score(X_test, test_pred_labels)

    train_silhouette_scores_em.append(silhouette_train)
    test_silhouette_scores_em.append(silhouette_test)

knee_train = KneeLocator(range(3, X_train.shape[1]), train_silhouette_scores_em, curve='convex', direction='decreasing')
em = GaussianMixture(n_components=knee_train.knee)
em.fit(X_train)
X_train = em.predict_proba(X_train)
X_test = em.predict_proba(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)

# bank (EM)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')


# Run EM clustering
dim_red = 'EM'
train_silhouette_scores_em = []
test_silhouette_scores_em = []
k_range = range(3, X_train.shape[1])

for i in k_range:
    em = GaussianMixture(n_components=i)
    em.fit(X_train)
    train_pred_labels = em.predict(X_train)
    test_pred_labels = em.predict(X_test)

    silhouette_train = silhouette_score(X_train, train_pred_labels)
    silhouette_test = silhouette_score(X_test, test_pred_labels)

    train_silhouette_scores_em.append(silhouette_train)
    test_silhouette_scores_em.append(silhouette_test)

knee_train = KneeLocator(range(3, X_train.shape[1]), train_silhouette_scores_em, curve='convex', direction='decreasing')
em = GaussianMixture(n_components=knee_train.knee)
em.fit(X_train)
X_train = em.predict_proba(X_train)
X_test = em.predict_proba(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)
