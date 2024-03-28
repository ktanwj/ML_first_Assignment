from utils import *
from kneed import KneeLocator
from sklearn.metrics import mean_squared_error

def plot_em_rp(X_train, X_test, data_lab):
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

    # generate RP
    reconstruction_errors = []

    # Try different number of components
    for n_components in range(1, X_train.shape[1] + 1):
        # Perform Random Projection
        rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        X_rp = rp.fit_transform(X_train)
        
        # Reconstruct data
        X_reconstructed = np.dot(X_rp, rp.components_)
        
        # Compute reconstruction error
        reconstruction_error = mean_squared_error(X_train, X_reconstructed)
        reconstruction_errors.append(reconstruction_error)
    
    knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')
    n_component_best = knee_train.knee

    # Reducing the dimensionality to the selected number of components
    rp = GaussianRandomProjection(n_components=n_component_best, random_state=42)
    X_train = rp.fit_transform(X_train)
    X_test = rp.fit_transform(X_test)

    # Run EM clustering on the reduced data
    train_silhouette_scores_rp = []
    test_silhouette_scores_rp = []
    k_range = range(2, 20)

    for i in k_range:
        gmm = GaussianMixture(n_components=i)
        gmm.fit(X_train)
        train_pred_labels = gmm.predict(X_train)
        test_pred_labels = gmm.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_rp.append(silhouette_train)
        test_silhouette_scores_rp.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_rp, marker='o', color='blue', label='Train, after RP')
    plt.plot(k_range, test_silhouette_scores_rp, marker='o', color='red', label='Test, after RP')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of component (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Expectation Maximization Silhouette score for {data_lab} before and after RP')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_em_rp_{data_lab}.png')


def plot_kmeans_rp(X_train, X_test, data_lab):
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

    # generate RP
    reconstruction_errors = []

    # Try different number of components
    for n_components in range(1, X_train.shape[1] + 1):
        # Perform Random Projection
        rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        X_rp = rp.fit_transform(X_train)
        
        # Reconstruct data
        X_reconstructed = np.dot(X_rp, rp.components_)
        
        # Compute reconstruction error
        reconstruction_error = mean_squared_error(X_train, X_reconstructed)
        reconstruction_errors.append(reconstruction_error)
    
    knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')
    n_component_best = knee_train.knee

    # Reducing the dimensionality to the selected number of components
    rp = GaussianRandomProjection(n_components=n_component_best, random_state=42)
    X_train = rp.fit_transform(X_train)
    X_test = rp.fit_transform(X_test)

    # Run k-means clustering on the reduced data
    train_silhouette_scores_rp = []
    test_silhouette_scores_rp = []
    k_range = range(2, 20)

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores_rp.append(silhouette_train)
        test_silhouette_scores_rp.append(silhouette_test)

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores_rp, marker='o', color='blue', label='Train, after RP')
    plt.plot(k_range, test_silhouette_scores_rp, marker='o', color='red', label='Test, after RP')
    plt.plot(k_range, train_silhouette_scores, linestyle="--", marker='x', color='blue', label='Train')
    plt.plot(k_range, test_silhouette_scores, linestyle="--",  marker='x', color='red', label='Test')

    plt.axvline(5, color="r", linestyle="--")
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhouette score for {data_lab} before and after RP')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_kmeans_rp_{data_lab}.png')


""" MAIN SCRIPT """
# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')
plot_kmeans_rp(X_train, X_test, data_lab)
plot_em_rp(X_train, X_test, data_lab)

# weather
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')
plot_kmeans_rp(X_train, X_test, data_lab)
plot_em_rp(X_train, X_test, data_lab)