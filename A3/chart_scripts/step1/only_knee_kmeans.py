from utils import *
from kneed import KneeLocator

"""
knee plot for bank
"""
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

def plot_silhouette(k_range, data_lab = 'Bank'):
    train_silhouette_scores = []
    test_silhouette_scores = []

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores.append(silhouette_train)
        test_silhouette_scores.append(silhouette_test)
    
    knee_train = KneeLocator(k_range, train_silhouette_scores, curve='concave', direction='increasing')

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores, marker='o', label='Train dataset')
    plt.plot(k_range, test_silhouette_scores, marker='x', label='Test dataset')
    plt.axvline(knee_train.knee, color="r", linestyle="--")
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhouette plot for {data_lab}')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_kmeans_{data_lab}.png')

k_range = range(3, 20)
plot_silhouette(k_range)

"""
knee plot for weather
"""
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

def plot_silhouette(k_range, data_lab = 'Weather'):
    train_silhouette_scores = []
    test_silhouette_scores = []

    for i in k_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        train_pred_labels = kmeans.labels_
        test_pred_labels = kmeans.predict(X_test)

        silhouette_train = silhouette_score(X_train, train_pred_labels)
        silhouette_test = silhouette_score(X_test, test_pred_labels)

        train_silhouette_scores.append(silhouette_train)
        test_silhouette_scores.append(silhouette_test)
    
    knee_train = KneeLocator(k_range, train_silhouette_scores, curve='concave', direction='increasing')

    # generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_silhouette_scores, marker='o', label='Train dataset')
    plt.plot(k_range, test_silhouette_scores, marker='x', label='Test dataset')
    plt.axvline(knee_train.knee, color="r", linestyle="--")
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(f'KMeans Silhouette plot for {data_lab}')
    plt.legend()
    plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_kmeans_{data_lab}.png')

k_range = range(3, 20)
plot_silhouette(k_range)