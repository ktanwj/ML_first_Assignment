from utils import *
from kneed import KneeLocator

def plot_em_pca(X_train, X_test, data_lab):
    # plot initial EM
    train_silhouette_scores = []
    test_silhouette_scores = []
    k_range = range(2, 20)

    for i in k_range:
        gmm = GaussianMixture(n_components=i)
        gmm.fit(X_train)
        train_pred_labels = gmm.predict(X_train)
        test_pred_labels = gmm.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores.append(silhouette_train)
        test_silhouette_scores.append(silhouette_test)

    # generate PCA
    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train)

    cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
    n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1

    # Reducing the dimensionality to the selected number of components
    pca = PCA(n_components=n_components_90)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    # Run k-means clustering on the reduced data
    train_silhouette_scores_pca = []
    test_silhouette_scores_pca = []
    k_range = range(2, 20)

    for i in k_range:
        gmm = GaussianMixture(n_components=i)
        gmm.fit(X_train)
        train_pred_labels = gmm.predict(X_train)
        test_pred_labels = gmm.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_pca.append(silhouette_train)
        test_silhouette_scores_pca.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_pca, marker='o', color='blue', label='Train, after PCA')
    plt.plot(k_range, test_silhouette_scores_pca, marker='o', color='red', label='Test, after PCA')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of component (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Expectation Maximization Silhouette score for {data_lab} before and after PCA')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_em_pca_{data_lab}.png')


def plot_kmeans_pca(X_train, X_test, data_lab):
    # plot initial Kmeans
    train_silhouette_scores = []
    test_silhouette_scores = []
    k_range = range(2, 20)

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores.append(silhouette_train)
        test_silhouette_scores.append(silhouette_test)

    # generate PCA
    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train)

    cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
    n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1

    # Reducing the dimensionality to the selected number of components
    pca = PCA(n_components=n_components_90)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    # Run k-means clustering on the reduced data
    train_silhouette_scores_pca = []
    test_silhouette_scores_pca = []
    k_range = range(2, 20)

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_pca.append(silhouette_train)
        test_silhouette_scores_pca.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_pca, marker='o', color='blue', label='Train, after PCA')
    plt.plot(k_range, test_silhouette_scores_pca, marker='o', color='red', label='Test, after PCA')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhouette score for {data_lab} before and after PCA')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_kmeans_pca_{data_lab}.png')


""" MAIN SCRIPT """
# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')
plot_kmeans_pca(X_train, X_test, data_lab)
plot_em_pca(X_train, X_test, data_lab)

# weather
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')
plot_kmeans_pca(X_train, X_test, data_lab)
plot_em_pca(X_train, X_test, data_lab)