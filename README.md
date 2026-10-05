# Forecasting Sticker Sales

Kaggle **Playground Series - Season 5, Episode 1** 데이터를 활용해 국가, 매장, 상품별 스티커 판매량(`num_sold`)을 예측한 시계열 회귀 프로젝트입니다.

🔗 [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e1/overview)

## 프로젝트 목표

날짜, 국가, 매장, 상품 정보를 활용하여 향후 스티커 판매량을 예측하는 머신러닝 모델을 구축했습니다.

## 분석 과정

### 1. 데이터 전처리
- `num_sold` 결측값 제거
- `date`를 `year`, `month`, `day` 파생변수로 변환
- `country`, `store`, `product`를 `LabelEncoder`로 인코딩
- 모델 학습에 불필요한 `id`, `date` 컬럼 제거

### 2. 모델링
- Train / Validation 데이터를 8:2 비율로 분리
- `XGBRegressor`를 활용해 판매량 예측 모델 학습

### 3. 평가

Kaggle 공식 평가 지표인 **MAPE (Mean Absolute Percentage Error)** 를 기준으로 모델 성능을 확인했습니다.

## 사용 기술

`Python` `Pandas` `NumPy`  
`Matplotlib` `Seaborn` `Scikit-learn` `XGBoost`

## Notebook

전체 EDA, 전처리 및 모델링 과정은  
`Forecasting_Sticker_Sales.ipynb`에서 확인할 수 있습니다.
