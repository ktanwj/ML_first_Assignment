from utils import *

# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (%)')
plt.title(f'Explained Variance Ratio by Principal Component for {data_lab} dataset')
plt.xticks(range(1, len(cumulative_explained_variance_ratio) + 1))

plt.axhline(y=90, color='gray', linestyle='--', label='90% Explained Variance')
n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1
plt.plot(n_components_90, 90, marker='o', markersize=8, color='red')
plt.annotate(f'Optimal PCA: {n_components_90}', xy=(n_components_90, 90),
             xytext=(n_components_90, 80),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'PCA_variance_{data_lab}.png')

# weather
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (%)')
plt.title(f'Explained Variance Ratio by Principal Component for {data_lab} dataset')
plt.xticks(range(1, len(cumulative_explained_variance_ratio) + 1))

plt.axhline(y=90, color='gray', linestyle='--', label='90% Explained Variance')
n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1
plt.plot(n_components_90, 90, marker='o', markersize=8, color='red')
plt.annotate(f'Optimal PCA: {n_components_90}', xy=(n_components_90, 90),
             xytext=(n_components_90, 80),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'PCA_variance_{data_lab}.png')