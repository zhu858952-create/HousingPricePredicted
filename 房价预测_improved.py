import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 加载数据 ====================
print('=' * 60)
print('step 1. load data')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_full = np.log1p(train['SalePrice'])
test_ids = test['Id']

X_full = train.drop(['Id', 'SalePrice'], axis=1)
X_test_orig = test.drop(['Id'], axis=1)

# ==================== 2. 数据预处理 ====================
print('\n' + '=' * 60)
print('step 2. data preprocessing')
all_data = pd.concat([X_full, X_test_orig], axis=0)

num_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())

cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    all_data[col] = all_data[col].fillna('None')

all_data = pd.get_dummies(all_data)
print(f'feature dim: {all_data.shape[1]}')

X_full_processed = all_data.iloc[:len(X_full), :]
X_test_processed = all_data.iloc[len(X_full):, :]

# ==================== 3. 划分训练/验证集（先划分，防泄露） ====================
print('\n' + '=' * 60)
print('step 3. split before scaling')
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full_processed, y_full, test_size=0.2, random_state=42
)

# ==================== 4. 标准化 ====================
print('\n' + '=' * 60)
print('step 4. scaling')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test_processed)

# ==================== 5. 基线交叉验证 ====================
print('\n' + '=' * 60)
print('step 5. baseline cv')
base_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.7, random_state=42
)
cv_scores = cross_val_score(
    base_model, X_train_scaled, y_train,
    cv=5, scoring='neg_root_mean_squared_error'
)
print(f'CV RMSE (log): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# ==================== 6. 网格搜索（不带早停） ====================
print('\n' + '=' * 60)
print('step 6. grid search')
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5],
    'subsample': [0.7, 0.8]
}

# 注意：基础模型不要设置 early_stopping_rounds
xgb_grid = xgb.XGBRegressor(
    n_estimators=1000, random_state=42, eval_metric='rmse'
)

grid_search = GridSearchCV(
    xgb_grid, param_grid,
    cv=5, scoring='neg_root_mean_squared_error',
    verbose=1, n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f'Best params: {grid_search.best_params_}')
print(f'Best CV RMSE (log): {-grid_search.best_score_:.4f}')

# ==================== 7. 用最佳参数+早停训练最终模型 ====================
print('\n' + '=' * 60)
print('step 7. final model with early stopping')
best_params = grid_search.best_params_
final_model = xgb.XGBRegressor(
    **best_params,
    n_estimators=2000,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
final_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_valid_scaled, y_valid)],
    verbose=100
)
print(f'Best iteration: {final_model.best_iteration}')

# ==================== 8. 预测并提交 ====================
print('\n' + '=' * 60)
print('step 8. predict and save')
preds_log = final_model.predict(X_test_scaled)
final_preds = np.expm1(preds_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
submission.to_csv('submission_improved.csv', index=False)
print('File saved: submission_improved.csv')

# ==================== 9. 验证集性能 ====================
val_preds_log = final_model.predict(X_valid_scaled)
val_rmse = np.sqrt(mean_squared_error(y_valid, val_preds_log))
print(f'Validation RMSE (log): {val_rmse:.4f}')