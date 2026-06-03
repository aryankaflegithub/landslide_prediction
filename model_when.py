import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("MODEL 2: WHEN - TEMPORAL TREND PREDICTION")

# Load data
train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')

print(f"DATA LOADED")
print(f"Train: {len(train_df)} events")
print(f"Test: {len(test_df)} events")

# Aggregate by month for temporal analysis
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

train_monthly = train_df.groupby(train_df['date'].dt.to_period('M')).agg({
    'OBJECTID': 'count',  # Number of events
    'month': 'first',
    'year': 'first'
}).reset_index()
train_monthly.columns = ['period', 'event_count', 'month', 'year']
train_monthly['date'] = train_monthly['period'].dt.to_timestamp()

test_monthly = test_df.groupby(test_df['date'].dt.to_period('M')).agg({
    'OBJECTID': 'count',
    'month': 'first',
    'year': 'first'
}).reset_index()
test_monthly.columns = ['period', 'event_count', 'month', 'year']
test_monthly['date'] = test_monthly['period'].dt.to_timestamp()

print(f"MONTHLY AGGREGATION")
print(f"Train months: {len(train_monthly)}")
print(f"Test months: {len(test_monthly)}")

# Engineer temporal features
def engineer_temporal_features(df):

    df = df.sort_values('date').reset_index(drop=True)

    # Lag features (past 1, 3, 6 months)
    df['lag_1m'] = df['event_count'].shift(1)
    df['lag_3m'] = df['event_count'].shift(3)
    df['lag_6m'] = df['event_count'].shift(6)

    # Rolling average (trend)
    df['avg_3m'] = df['event_count'].shift(1).rolling(3, min_periods=1).mean()
    df['avg_6m'] = df['event_count'].shift(1).rolling(6, min_periods=1).mean()

    # Cyclical encoding for month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # Seasonal indicator (monsoon: June-Sept)
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    return df

train_monthly = engineer_temporal_features(train_monthly)
test_monthly = engineer_temporal_features(test_monthly)

# Features for WHEN model
temporal_features = ['lag_1m', 'lag_3m', 'lag_6m', 'avg_3m', 'avg_6m',
                     'sin_month', 'cos_month', 'is_monsoon']

# Remove NaN rows (from lag/rolling features)
train_valid = train_monthly.dropna()
test_valid = test_monthly.dropna()

X_train = train_valid[temporal_features]
X_test = test_valid[temporal_features]
y_train = train_valid['event_count']
y_test = test_valid['event_count']

print(f"FEATURES ENGINEERED")
print(f"Train rows after NA removal: {len(X_train)}")
print(f"Test rows after NA removal: {len(X_test)}")

# Train Gradient Boosting Regressor
print(f"TRAINING TEMPORAL MODEL")
model_when = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

model_when.fit(X_train, y_train)

# Evaluate
y_pred = model_when.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MODEL PERFORMANCE")
print(f"MAE (Mean Absolute Error): {mae:.2f} events/month")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} events/month")
print(f"R2 Score: {r2:.3f}")

# Feature importance
print(f"FEATURE IMPORTANCE")
for feat, imp in zip(temporal_features, model_when.feature_importances_):
    print(f"  {feat:15s}: {imp:.3f}")

# Save model
with open('model_when.pkl', 'wb') as f:
    pickle.dump(model_when, f)
print(f"MODEL SAVED: model_when.pkl")

# Plot predictions vs actual
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Time series
test_plot = test_valid.copy()
test_plot['predicted'] = y_pred
test_plot = test_plot.sort_values('date')

axes[0].plot(test_plot['date'], test_plot['event_count'], marker='o',
             label='Actual', linewidth=2, markersize=4)
axes[0].plot(test_plot['date'], test_plot['predicted'], marker='s',
             label='Predicted', linewidth=2, markersize=4, alpha=0.7)
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Event Count', fontsize=11)
axes[0].set_title('WHEN Model - Events Over Time', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test.values - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Count', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title('Residual Plot - WHEN Model', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_when_results.png', dpi=150, bbox_inches='tight')
print(f"RESULTS SAVED: model_when_results.png")

print(f"MODEL 2 COMPLETE")

