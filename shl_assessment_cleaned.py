import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # for saving the model


# Load train, test, and sample submission files
train = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/train.csv')
test = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/test.csv')
sample_submission = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/sample_submission.csv')

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print(train.columns)

# Output:
# Train Shape: (444, 2)
# Test Shape: (195, 1)
# Index(['filename', 'label'], dtype='object')


print(train[['filename', 'label']])


# Output:
#            filename  label
# 0    audio_1261.wav    1.0
# 1     audio_942.wav    1.5
# 2    audio_1110.wav    1.5
# 3    audio_1024.wav    1.5
# 4     audio_538.wav    2.0
# ..              ...    ...
# 439   audio_494.wav    5.0
# 440   audio_363.wav    5.0
# 441   audio_481.wav    5.0
# 442   audio_989.wav    5.0
# 443  audio_1163.wav    5.0
# 
# [444 rows x 2 columns]


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
        'rmse': np.mean(librosa.feature.rms(y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'mfcc_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0).mean(),
        'mfcc_std': np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0).std(),
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'tonnetz': np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)),
        'tempo': librosa.beat.tempo(y, sr=sr)[0]
    }
    return features



from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_



from sklearn.metrics import mean_squared_error

# Predictions on train and test data
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Evaluation
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_preds)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred)))


# Output:
# Train RMSE: 0.3831661500416683
# Test RMSE: 0.9798677265570415


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_test)



from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)


# Output:
# GradientBoostingRegressor(learning_rate=0.05, max_depth=5, min_samples_leaf=2,
#                           min_samples_split=5, n_estimators=500,
#                           random_state=42)


from sklearn.metrics import mean_squared_error
import numpy as np

train_preds = gbr.predict(X_train_scaled)
val_preds = gbr.predict(X_val_scaled)

print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_preds)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred)))


# Output:
# Train RMSE: 0.047326225549307994
# Test RMSE: 0.9798677265570415


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



train_features = []

for file in tqdm(train['filename']):  # or 'file' based on your column name
    file_path = f'/kaggle/input/shl-intern-hiring-assessment/dataset/audios_train/{file}'
    features = extract_features(file_path)
    train_features.append(features)

train_features_df = pd.DataFrame(train_features)
train_df = pd.concat([train, train_features_df], axis=1)

# Output:
# 100%|██████████| 444/444 [04:44<00:00,  1.56it/s]


test_features = []

for file in tqdm(test['filename']):
    file_path = f'/kaggle/input/shl-intern-hiring-assessment/dataset/audios_test/{file}'  # Added /
    features = extract_features(file_path)
    test_features.append(features)

test_features_df = pd.DataFrame(test_features)

# Output:
# 100%|██████████| 195/195 [01:59<00:00,  1.63it/s]


X = train_df.drop(['filename', 'label'], axis=1)  # Features
y = train_df['label']  # Target/Labels
X_test = test_features_df  # Test Data Features


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Output:
# RandomForestRegressor(n_estimators=300, random_state=42)


y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print("Validation MSE:", mse)

# Output:
# Validation MSE: 0.960140761548065


test_preds = model.predict(X_test)


sample_submission['label'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


# Provide path of your input audio file
audio_path = '/kaggle/input/shl-intern-hiring-assessment/dataset/audios_test/audio_1159.wav'  # upload your audio file in working directory

# Extract features
features = extract_features(audio_path)

# Convert to DataFrame
input_df = pd.DataFrame([features])

# Predict Grammar Score
predicted_score = model.predict(input_df)[0]

print("Predicted Grammar Score: ", predicted_score)


# Output:
# Predicted Grammar Score:  3.72


joblib.dump(model, 'grammar_score_model.pkl')


# Output:
# ['grammar_score_model.pkl']


# Predict Grammar Score for test data
test_predictions = model.predict(test_features_df)



# Predict on Training Data
train_predictions = model.predict(X)

# Compare Actual vs Predicted
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=y, y=train_predictions)
plt.xlabel('Actual Grammar Score')
plt.ylabel('Predicted Grammar Score')
plt.title('Actual vs Predicted Grammar Score')
plt.show()


# Output:
# <Figure size 800x600 with 1 Axes>


from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y, train_predictions))
print("Train RMSE:", rmse)

# Output:
# Train RMSE: 0.5565836733011357


sample_submission = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/dataset/sample_submission.csv')



sample_submission['score'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)



import pandas as pd

# Load your submission file
submission = pd.read_csv('/kaggle/working/submission.csv')

# View the first few rows
submission.head()


# Output:
#          filename  label     score
# 0   audio_706.wav      0  3.255000
# 1   audio_800.wav      0  3.611667
# 2    audio_68.wav      0  3.761667
# 3  audio_1267.wav      0  3.755000
# 4   audio_683.wav      0  3.581667


