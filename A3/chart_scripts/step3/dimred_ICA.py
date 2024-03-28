from utils import *
from kneed import KneeLocator

def plot_em_ica(X_train, X_test, data_lab):
    # plot initial em
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

    # generate ICA
    ica = FastICA(n_components=X_train.shape[1])
    X_transformed = ica.fit_transform(X_train)
    
    # calc kurtosis
    component_kurtosis = kurtosis(X_transformed)

    # choose number of component based on max kurtosis
    n_component_best = np.argmax(component_kurtosis) + 1

    # Reducing the dimensionality to the selected number of components
    ica = FastICA(n_components=n_component_best)
    X_train = ica.fit_transform(X_train)
    X_test = ica.fit_transform(X_test)

    # Run k-means clustering on the reduced data
    train_silhouette_scores_ica = []
    test_silhouette_scores_ica = []
    k_range = range(2, 20)

    for i in k_range:
        gmm = GaussianMixture(n_components=i)
        gmm.fit(X_train)
        train_pred_labels = gmm.predict(X_train)
        test_pred_labels = gmm.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_ica.append(silhouette_train)
        test_silhouette_scores_ica.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_ica, marker='o', color='blue', label='Train, after ICA')
    plt.plot(k_range, test_silhouette_scores_ica, marker='o', color='red', label='Test, after ICA')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of component (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Expectation Maximization Silhouette score for {data_lab} before and after ICA')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_em_ica_{data_lab}.png')


def plot_kmeans_ica(X_train, X_test, data_lab):
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

    # generate ICA
    ica = FastICA(n_components=X_train.shape[1])
    X_transformed = ica.fit_transform(X_train)
    
    # calc kurtosis
    component_kurtosis = kurtosis(X_transformed)

    # choose number of component based on max kurtosis
    n_component_best = np.argmax(component_kurtosis) + 1

    # Reducing the dimensionality to the selected number of components
    ica = FastICA(n_components=n_component_best)
    X_train = ica.fit_transform(X_train)
    X_test = ica.fit_transform(X_test)

    # Run k-means clustering on the reduced data
    train_silhouette_scores_ica = []
    test_silhouette_scores_ica = []
    k_range = range(2, 20)

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_ica.append(silhouette_train)
        test_silhouette_scores_ica.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_ica, marker='o', color='blue', label='Train, after ICA')
    plt.plot(k_range, test_silhouette_scores_ica, marker='o', color='red', label='Test, after ICA')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhouette score for {data_lab} before and after ICA')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_kmeans_ica_{data_lab}.png')


""" MAIN SCRIPT """
# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')
plot_kmeans_ica(X_train, X_test, data_lab)
plot_em_ica(X_train, X_test, data_lab)

# weather
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')
plot_kmeans_ica(X_train, X_test, data_lab)
plot_em_ica(X_train, X_test, data_lab)