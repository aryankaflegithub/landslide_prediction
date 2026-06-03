import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("MODEL 1: WHERE - SPATIAL RISK PREDICTION")

# Load data
train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')

print(f"DATA LOADED")
print(f"Train: {len(train_df)} events")
print(f"Test: {len(test_df)} events")

# Create spatial grid (0.1 degree cells = ~10km)
GRID_SIZE = 0.1

def get_grid_cell(lon, lat, grid_size=GRID_SIZE):

    grid_lon = int(np.floor(lon / grid_size) * grid_size * 10000)
    grid_lat = int(np.floor(lat / grid_size) * grid_size * 10000)
    return f"{grid_lon}_{grid_lat}"

# Assign grid cells
train_df['grid_cell'] = train_df.apply(lambda row: get_grid_cell(row['lon'], row['lat']), axis=1)
test_df['grid_cell'] = test_df.apply(lambda row: get_grid_cell(row['lon'], row['lat']), axis=1)

print(f"GRID ANALYSIS")
print(f"Unique cells in train: {train_df['grid_cell'].nunique()}")
print(f"Unique cells in test: {test_df['grid_cell'].nunique()}")

# Features for WHERE model: location + spatial patterns
spatial_features = ['lon', 'lat', 'spatial_density']

X_train = train_df[spatial_features].fillna(0)
X_test = test_df[spatial_features].fillna(0)

# Create binary target: High risk (1) vs Low risk (0)
# High risk = cells with > median events
median_events_per_cell = train_df['spatial_density'].median()
y_train = (train_df['spatial_density'] > median_events_per_cell).astype(int)
y_test = (test_df['spatial_density'] > median_events_per_cell).astype(int)

print(f"TARGET DISTRIBUTION")
print(f"Train - High risk: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Train - Low risk: {(1-y_train).sum()} ({(1-y_train).mean()*100:.1f}%)")
print(f"Test - High risk: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"Test - Low risk: {(1-y_test).sum()} ({(1-y_test).mean()*100:.1f}%)")

# Train Random Forest
print(f"TRAINING SPATIAL MODEL...")
model_where = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model_where.fit(X_train, y_train)

# Evaluate
y_pred = model_where.predict(X_test)
y_pred_proba = model_where.predict_proba(X_test)[:, 1]

accuracy = (y_pred == y_test).mean()
auc = roc_auc_score(y_test, y_pred_proba)

print(f"MODEL PERFORMANCE")
print(f" Accuracy: {accuracy:.3f}")
print(f" AUC Score: {auc:.3f}")

print(f"CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# Feature importance
print(f"FEATURE IMPORTANCE")
for feat, imp in zip(spatial_features, model_where.feature_importances_):
    print(f"  {feat:20s}: {imp:.3f}")

# Save model
with open('model_where.pkl', 'wb') as f:
    pickle.dump(model_where, f)
print(f"MODEL SAVED: model_where.pkl")

# Create risk map
print(f"CREATING RISK MAP...")
lon_range = np.arange(test_df['lon'].min() - 0.5, test_df['lon'].max() + 0.5, GRID_SIZE)
lat_range = np.arange(test_df['lat'].min() - 0.5, test_df['lat'].max() + 0.5, GRID_SIZE)

risk_map = []
for lon in lon_range:
    for lat in lat_range:
        risk = model_where.predict_proba([[lon, lat, 0]])[0, 1]
        risk_map.append({'lon': lon, 'lat': lat, 'risk': risk})

risk_df = pd.DataFrame(risk_map)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Risk map
scatter = axes[0].scatter(risk_df['lon'], risk_df['lat'], c=risk_df['risk'],
                         cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Longitude', fontsize=11)
axes[0].set_ylabel('Latitude', fontsize=11)
axes[0].set_title('Spatial Risk Map - Nepal', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[0], label='Risk Score (0-1)')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title('ROC Curve - WHERE Model', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_where_results.png', dpi=150, bbox_inches='tight')
print(f"RESULTS SAVED: model_where_results.png")

print(f"MODEL 1 COMPLETE")

