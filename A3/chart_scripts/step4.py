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
# bank tuning (ICA)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# ICA
dim_red = 'ICA'
ica = FastICA(n_components=X_train.shape[1])
X_transformed = ica.fit_transform(X_train)
component_kurtosis = kurtosis(X_transformed)
n_component_best = np.argmax(component_kurtosis) + 1
ica = FastICA(n_components=n_component_best)
X_train = ica.fit_transform(X_train)
X_test = ica.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)

# weather (ICA)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

dim_red = 'ICA'
ica = FastICA(n_components=X_train.shape[1])
X_transformed = ica.fit_transform(X_train)
component_kurtosis = kurtosis(X_transformed)
n_component_best = np.argmax(component_kurtosis) + 1
ica = FastICA(n_components=n_component_best)
X_train = ica.fit_transform(X_train)
X_test = ica.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)

"""
PCA
"""
# bank tuning (PCA)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')

# PCA
dim_red = 'PCA'
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1

# Reducing the dimensionality to the selected number of components
pca = PCA(n_components=n_components_90)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)

# weather tuning (PCA)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')

dim_red = 'PCA'
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
n_components_90 = np.argmax(cumulative_explained_variance_ratio > 90) + 1

# Reducing the dimensionality to the selected number of components
pca = PCA(n_components=n_components_90)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)


"""
RP
"""
# bank tuning (RP)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')
dim_red = 'RP'

reconstruction_errors = []

for n_components in range(1, X_train.shape[1] + 1):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X_train)    
    X_reconstructed = np.dot(X_rp, rp.components_)
    reconstruction_error = mean_squared_error(X_train, X_reconstructed)
    reconstruction_errors.append(reconstruction_error)

knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')
n_component_best = knee_train.knee
rp = GaussianRandomProjection(n_components=n_component_best, random_state=42)
X_train = rp.fit_transform(X_train)
X_test = rp.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)

# weather tuning (RP)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')
dim_red = 'RP'

reconstruction_errors = []

for n_components in range(1, X_train.shape[1] + 1):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X_train)    
    X_reconstructed = np.dot(X_rp, rp.components_)
    reconstruction_error = mean_squared_error(X_train, X_reconstructed)
    reconstruction_errors.append(reconstruction_error)

knee_train = KneeLocator(range(1, X_train.shape[1] + 1), reconstruction_errors, curve='convex', direction='decreasing')
n_component_best = knee_train.knee
rp = GaussianRandomProjection(n_components=n_component_best, random_state=42)
X_train = rp.fit_transform(X_train)
X_test = rp.fit_transform(X_test)

train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)



"""
ISO
"""
# bank tuning (ISO)
data_lab = 'Bank'
X_test = get_pickle_file(BANK_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(BANK_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(BANK_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(BANK_FOLDER_PATH+'y_train.pkl')
dim_red = 'ISO'
silhouette_scores = []
neighbor_range = range(2, X_train.shape[1])
# Try different numbers of neighbors
for n_neighbors in neighbor_range:
    # Perform Isomap
    isomap = Isomap(n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X_train)
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X_isomap, y_train))

n_component_best = np.argmax(silhouette_scores) + 1

# Reducing the dimensionality to the selected number of components
iso = Isomap(n_neighbors=n_component_best)
X_train = iso.fit_transform(X_train)
X_test = iso.fit_transform(X_test)
train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='bank',scores=['f1'], dim_red=dim_red)


# weather tuning (ISO)
data_lab = 'Weather'
X_test = get_pickle_file(WEATHER_FOLDER_PATH+'x_test.pkl')
X_train = get_pickle_file(WEATHER_FOLDER_PATH+'x_train.pkl')
y_test = get_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl')
y_train = get_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl')
dim_red = 'ISO'
silhouette_scores = []
neighbor_range = range(2, X_train.shape[1])
# Try different numbers of neighbors
for n_neighbors in neighbor_range:
    # Perform Isomap
    isomap = Isomap(n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X_train)
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X_isomap, y_train))

n_component_best = np.argmax(silhouette_scores) + 1

# Reducing the dimensionality to the selected number of components
iso = Isomap(n_neighbors=n_component_best)
X_train = iso.fit_transform(X_train)
X_test = iso.fit_transform(X_test)
train_validation_report, test_report = model_tuning(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_name='weather',scores=['f1'], dim_red=dim_red)