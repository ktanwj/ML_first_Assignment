from utils import *

# bank data
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

transformer = FastICA(n_components=X_train.shape[1],
                      random_state=0,
                      whiten='unit-variance')
X_transformed = transformer.fit_transform(X_train)
component_kurtosis = kurtosis(X_transformed)
abs_avg_kurtosis = np.mean(np.abs(component_kurtosis))

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(component_kurtosis) + 1), component_kurtosis, alpha=0.7)
plt.axhline(y=abs_avg_kurtosis, color='gray', linestyle='--', label='Average Kurtosis')

plt.xlabel('Component')
plt.ylabel('Kurtosis')
plt.title(f'Kurtosis by Independent component for {data_lab} dataset')
plt.xticks(range(1, len(component_kurtosis) + 1))
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'ICA_kurtosis_{data_lab}.png')

# weather data
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

transformer = FastICA(n_components=X_train.shape[1],
                      random_state=0,
                      whiten='unit-variance')
X_transformed = transformer.fit_transform(X_train)
component_kurtosis = kurtosis(X_transformed)
abs_avg_kurtosis = np.mean(np.abs(component_kurtosis))

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(component_kurtosis) + 1), component_kurtosis, alpha=0.7)
plt.axhline(y=abs_avg_kurtosis, color='gray', linestyle='--', label='Average Kurtosis')

plt.xlabel('Component')
plt.ylabel('Kurtosis')
plt.title(f'Kurtosis by Independent component for {data_lab} dataset')
plt.xticks(range(1, len(component_kurtosis) + 1))
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'ICA_kurtosis_{data_lab}.png')