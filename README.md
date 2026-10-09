# 스티커 판매량 예측

## 분석 질문과 방법

판매량(num_sold)을 대상으로 개별 절대 상대오차 평균인 MAPE를 계산합니다. 실제값 0의 상대오차는 정의되지 않아 개수와 MAE를 함께 보고합니다. 2015년 선택용, 2016년 최종 검증으로 미래 예측 상황을 반영합니다. 전년 동일 월일 기준 모델과 XGBoost를 비교하며 마지막 연도 성능으로 모델을 재선택하지 않습니다. 타깃 결측 제외 편향과 2017~2019년 장기 외삽 한계를 명시합니다.

## 실행

Python 3.11 이상에서 저장소 폴더를 작업 디렉터리로 사용합니다.

```bash
python -m pip install -r requirements.txt
python analysis.py
```

[Forecasting_Sticker_Sales.ipynb](Forecasting_Sticker_Sales.ipynb)에서 실행 결과와 그래프를 확인할 수 있습니다.
원본 데이터 경로는 기존 저장소와 동일합니다. 주가 프로젝트만 최초 실행 시 Yahoo Finance 연결이 필요합니다.

## 결과와 한계

실제 실행 결과는 `outputs/metrics.json`과 `outputs/`의 비교표·그래프에 저장됩니다.
수정 전 저장된 점수는 새 검증 결과와 혼용하지 않습니다. 검증 점수는 대회 리더보드 점수가 아닙니다.
모델을 정한 뒤 제출 데이터 전체를 예측하며, 제출 파일을 만들었다는 사실이 대회에 제출했다는 의미는 아닙니다.
분석에서 확인한 관계와 제안은 실제 업무 개선 효과를 증명하지 않습니다.

## 재실행 결과 (2026-10-10)

```json
{
  "selected_model": "Seasonal baseline",
  "selection_year": 2015,
  "final_validation_year": 2016,
  "final_metrics": {
    "year": 2016,
    "model": "Seasonal baseline",
    "mape_percent": 12.688716050880789,
    "mae": 88.28800849417344,
    "zero_actual_rows": 0,
    "rows": 31767
  }
}
```

MAPE 기준으로 전년 동일 월일 모델을 선택했습니다. MAE에서는 XGBoost가 더 좋았으므로 최적 모델은 의사결정의 비용과 평가지표에 따라 달라집니다. 미래 제출 기간은 더 길어 성능이 달라질 수 있습니다.
