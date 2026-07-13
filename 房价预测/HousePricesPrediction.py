import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# ========== 1. 加载数据 ==========
print('=' * 60)
print('step 1. load data')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ========== 2. 数据预处理 & One-Hot 编码 ==========
print('\n' + '=' * 60)
print('step 2. data preprocessing')
y_train_full = np.log1p(train['SalePrice'])
test_ids = test['Id']

X_train_full = train.drop(['Id', 'SalePrice'], axis=1)
X_test = test.drop(['Id'], axis=1)

all_data = pd.concat([X_train_full, X_test], axis=0)

# 填充缺失值
num_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())

cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    all_data[col] = all_data[col].fillna('None')

# 执行 One-Hot 编码：解决字符串无法计算的问题
all_data = pd.get_dummies(all_data)
print(f'独热编码后特征维度: {all_data.shape}')

X_train_processed = all_data.iloc[:len(X_train_full), :]
X_test_processed = all_data.iloc[len(X_train_full):, :]

# ========== 3. 数据标准化 ==========
print('\n' + '=' * 60)
print('step 3. standardize data')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# ========== 4. 划分数据集 ==========
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_scaled, y_train_full, test_size=0.2, random_state=42
)

# ========== 5. 训练 XGBoost (适配 3.2.0 API) ==========
print('\n' + '=' * 60)
print('step 5. train XGBoost regressor')

# 【关键改动】：将 early_stopping_rounds 和 eval_metric 放到这里
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    # 新版本写法：在这里定义早停和评价标准
    early_stopping_rounds=50,
    eval_metric='rmse'
)

# fit 变得非常清爽，只需要传数据和验证集
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100
)

print(f'\n最佳迭代轮数: {xgb_model.best_iteration}')

# ========== 6. 预测与生成文件 ==========
preds_log = xgb_model.predict(X_test_scaled)
final_preds = np.expm1(preds_log) # 还原对数

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
submission.to_csv('submission_v3.csv', index=False)
print("恭喜！代码运行成功，文件已生成。")