from utils import *

# ISO MAP
"""
bank data
"""
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# Choose the range of neighbors to explore
neighbor_range = range(2, X_train.shape[1])

# Initialize an empty list to store silhouette scores
silhouette_scores = []

# Try different numbers of neighbors
for n_neighbors in neighbor_range:
    # Perform Isomap
    isomap = Isomap(n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X_train)
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X_isomap, y_train))

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(neighbor_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors')
plt.ylabel('Silhouette Score')
plt.title(f'Silhouette Score by Number of Neighbors (Isomap) for {data_lab} data')
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'/iso_silhouette_{data_lab}.png')

# ISO MAP
"""
bank data
"""
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

# Choose the range of neighbors to explore
neighbor_range = range(2, X_train.shape[1])

# Initialize an empty list to store silhouette scores
silhouette_scores = []

# Try different numbers of neighbors
for n_neighbors in neighbor_range:
    # Perform Isomap
    isomap = Isomap(n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X_train)
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X_isomap, y_train))

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(neighbor_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors')
plt.ylabel('Silhouette Score')
plt.title(f'Silhouette Score by Number of Neighbors (Isomap) for {data_lab} data')
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'/iso_silhouette_{data_lab}.png')