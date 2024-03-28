from utils import *
from kneed import KneeLocator

# bank
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# Random Projection
from sklearn.metrics import mean_squared_error
reconstruction_errors_runs = []
n_run = 10

for i in range(n_run):
    reconstruction_errors = []

    # Try different number of components
    for n_components in range(1, X_train.shape[1] + 1):
        # Perform Random Projection
        rp = GaussianRandomProjection(n_components=n_components)
        X_rp = rp.fit_transform(X_train)
        
        X_reconstructed = np.dot(X_rp, rp.components_)
        
        reconstruction_error = mean_squared_error(X_train, X_reconstructed)
        reconstruction_errors.append(reconstruction_error)

    reconstruction_errors_runs.append(reconstruction_errors)

averages = [sum(column) / len(column) for column in zip(*reconstruction_errors_runs)]
std_devs = [np.std(column) for column in zip(*reconstruction_errors_runs)]

knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')

plt.figure(figsize=(10, 6))
plt.plot(range(1, X_train.shape[1] + 1), averages, marker='o', linestyle='-')
for i in range(1, X_train.shape[1]):
    plt.fill_between([i], averages[i] - std_devs[i], averages[i] + std_devs[i], color='blue', alpha=0.5)
plt.axvline(knee_train.knee, color="r", linestyle="--")
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title(f'Reconstruction Error by Number of Components (RP - {data_lab})')
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_rp_{data_lab}.png')


# bank
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

# Random Projection
from sklearn.metrics import mean_squared_error
reconstruction_errors_runs = []
n_run = 10

for i in range(n_run):
    reconstruction_errors = []

    # Try different number of components
    for n_components in range(1, X_train.shape[1] + 1):
        # Perform Random Projection
        rp = GaussianRandomProjection(n_components=n_components)
        X_rp = rp.fit_transform(X_train)
        
        X_reconstructed = np.dot(X_rp, rp.components_)
        
        reconstruction_error = mean_squared_error(X_train, X_reconstructed)
        reconstruction_errors.append(reconstruction_error)

    reconstruction_errors_runs.append(reconstruction_errors)

averages = [sum(column) / len(column) for column in zip(*reconstruction_errors_runs)]
std_devs = [np.std(column) for column in zip(*reconstruction_errors_runs)]

knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')

plt.figure(figsize=(10, 6))
plt.plot(range(1, X_train.shape[1] + 1), averages, marker='o', linestyle='-')
for i in range(1, X_train.shape[1]):
    plt.fill_between([i], averages[i] - std_devs[i], averages[i] + std_devs[i], color='blue', alpha=0.5)
plt.axvline(knee_train.knee, color="r", linestyle="--")
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title(f'Reconstruction Error by Number of Components (RP - {data_lab})')
plt.grid(True)
plt.savefig(OUTPUT_FOLDER_PATH+f'/knee_rp_{data_lab}.png')