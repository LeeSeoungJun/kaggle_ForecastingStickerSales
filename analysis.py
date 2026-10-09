from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
    if font in fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
SEED = 2026

def preprocessing(frame):
    categorical = frame.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numerical = [c for c in frame if c not in categorical]
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), numerical),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])

def save_submission(sample, ids, target, prediction):
    assert list(sample.columns) == [ids.name, target], '제출 열 확인 필요'
    assert len(ids) == len(prediction) == len(sample)
    assert sample[ids.name].astype(str).tolist() == ids.astype(str).tolist(), '제출 ID 순서 불일치'
    assert np.isfinite(prediction).all()
    result = sample.copy()
    result[target] = prediction
    result.to_csv(OUT / 'submission.csv', index=False)
    return result

# %% 시계열 데이터 품질과 지표
from sklearn.base import clone
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])
sample = pd.read_csv('sample_submission.csv')
TARGET = 'num_sold'
KEYS = ['country', 'store', 'product']
assert train['id'].is_unique and test['id'].is_unique
assert not train.duplicated(['date'] + KEYS).any()
assert train.date.notna().all() and test.date.notna().all()
assert train.date.max() < test.date.min()
missing = train.groupby(KEYS).agg(rows=(TARGET, 'size'), observed=(TARGET, 'count'))
missing['missing_rate'] = 1 - missing.observed / missing.rows
display(missing.sort_values('missing_rate', ascending=False).head(15))
missing.to_csv(OUT / 'target_missingness.csv')
print('타깃 결측 제외:', train[TARGET].isna().sum())
train = train.dropna(subset=[TARGET]).sort_values('date').copy()
assert train[TARGET].ge(0).all()

def mape_percent(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    # 실제값 0에서는 상대오차가 정의되지 않으므로, 숨기지 않고 개수를 함께 보고한다.
    if np.any(actual == 0):
        return float('nan')
    return float(np.mean(np.abs(actual - pred) / np.abs(actual)) * 100)

def features(frame):
    result = frame[KEYS].copy()
    dates = frame['date']
    result['year'] = dates.dt.year
    result['month'] = dates.dt.month
    result['weekday'] = dates.dt.dayofweek
    result['day'] = dates.dt.day
    result['annual_sin'] = np.sin(2*np.pi*dates.dt.dayofyear/365.25)
    result['annual_cos'] = np.cos(2*np.pi*dates.dt.dayofyear/365.25)
    return result

def seasonal_baseline(history, future):
    # 동일 국가·매장·품목의 직전 연도 동일 월일. 결측 조합은 과거 그룹 평균으로 대체.
    lookup = history.set_index(KEYS + ['date'])[TARGET]
    keys = future[KEYS].copy()
    keys['date'] = future.date - pd.DateOffset(years=1)
    pred = lookup.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    groups = history.groupby(KEYS)[TARGET].mean()
    fallback = groups.reindex(pd.MultiIndex.from_frame(future[KEYS])).fillna(history[TARGET].mean()).to_numpy()
    return np.where(np.isnan(pred), fallback, pred)

model = Pipeline([('prep', preprocessing(features(train))),
                  ('model', XGBRegressor(n_estimators=250, max_depth=6, learning_rate=.05,
                                        random_state=SEED, n_jobs=4, tree_method='hist'))])
# %% 시간 순서 검증: 2015년은 선택용, 2016년은 최종 평가용
latest_year = int(train.date.dt.year.max())
rows = []
for year in [latest_year - 1, latest_year]:
    history = train.loc[train.date < pd.Timestamp(year=year, month=1, day=1)]
    hold = train.loc[train.date.dt.year.eq(year)]
    assert len(history) and len(hold) and history.date.max() < hold.date.min()
    fitted = clone(model).fit(features(history), history[TARGET])
    predictions = {'Seasonal baseline': seasonal_baseline(history, hold),
                   'XGBoost': np.maximum(fitted.predict(features(hold)), 0)}
    for name, pred in predictions.items():
        rows.append({'year': year, 'model': name, 'mape_percent': mape_percent(hold[TARGET], pred),
                     'mae': mean_absolute_error(hold[TARGET], pred),
                     'zero_actual_rows': int(hold[TARGET].eq(0).sum()), 'rows': len(hold)})
    if year == latest_year - 1:
        # 오직 선택용 연도 성능으로 제출 모델 결정. 실제값 0이 있으면 MAE 기준을 사용.
        ranking = pd.DataFrame(rows)
        criterion = 'mae' if hold[TARGET].eq(0).any() else 'mape_percent'
        selected = ranking.sort_values(criterion).iloc[0]['model']
    else:
        final_hold = hold.copy()
        final_hold['prediction'] = predictions[selected]
scores = pd.DataFrame(rows)
display(scores)
scores.to_csv(OUT / 'temporal_validation.csv', index=False)
print('선택용 연도에서 결정된 모델:', selected)
final_hold['absolute_error'] = abs(final_hold[TARGET] - final_hold.prediction)
final_hold['ape_percent'] = np.where(final_hold[TARGET] > 0, 100*final_hold.absolute_error/final_hold[TARGET], np.nan)
summary = final_hold.groupby(KEYS).agg(rows=(TARGET, 'size'), mae=('absolute_error', 'mean'), mape_percent=('ape_percent', 'mean'))
display(summary.sort_values('mae', ascending=False).head(15))
summary.to_csv(OUT / 'errors_by_series.csv')
final_hold.to_csv(OUT / 'holdout_predictions.csv', index=False)
actual_monthly = final_hold.groupby(final_hold.date.dt.to_period('M'))[[TARGET, 'prediction']].sum()
actual_monthly.index = actual_monthly.index.astype(str)
ax = actual_monthly.plot(figsize=(10, 5), title='최종 검증 연도 월별 판매량: 실제와 예측')
ax.set(ylabel='판매 수량', xlabel='월'); plt.tight_layout(); plt.savefig(OUT / 'monthly_validation.png'); plt.show(); plt.close()
metrics = {'selected_model': selected, 'selection_year': latest_year-1, 'final_validation_year': latest_year,
           'final_metrics': scores.loc[(scores.year == latest_year) & (scores.model == selected)].iloc[0].to_dict()}
(OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
# %% 전체 과거 데이터 학습 / 미래 제출
if selected == 'XGBoost':
    model.fit(features(train), train[TARGET])
    pred = np.maximum(model.predict(features(test)), 0)
else:
    pred = seasonal_baseline(train, test)
submission = save_submission(sample, test['id'], TARGET, pred)
display(submission.head())
