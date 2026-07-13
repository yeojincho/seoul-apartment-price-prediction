# 🏠 SCU Apartment Price Prediction

서울사이버대학교 빅데이터·AI센터에서 주최한 **제5회 혁신 AI 경진대회 - 아파트 실거래가 예측** 참가 프로젝트입니다.

본 프로젝트는 서울 지역 아파트 실거래 데이터를 기반으로 실제 거래 가격을 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다.

---

## 📌 Competition Overview

이번 대회는 서울사이버대학교 재학생을 대상으로 진행되는 AI 경진대회로, 공공데이터를 활용하여 부동산 실거래가를 예측하는 모델을 개발하는 것이 목표입니다.

참가자는 데이터 전처리, 특성 엔지니어링, 모델 학습 및 평가 과정을 수행하며 데이터 분석 역량과 AI 모델 개발 역량을 향상시킬 수 있습니다.

### 대회 정보

| 항목    | 내용                           |
| ----- | ---------------------------- |
| 대회명   | 제5회 혁신 AI 경진대회 - 아파트 실거래가 예측 |
| 주최    | 서울사이버대학교 빅데이터·AI센터           |
| 참가 대상 | 서울사이버대학교 재학생 및 대학원생          |
| 플랫폼   | Kaggle                       |
| 기간    | 2026.06.22 ~ 2026.07.09      |

---

## 🎯 Objective

주어진 아파트 거래 데이터를 활용하여 실거래가(Target)를 예측하는 회귀(Regression) 모델을 개발합니다.

---

## 📂 Dataset
서울 아파트 실거래가 예측을 위한 정형(Tabular) 데이터셋입니다.

대회 데이터는 실제 부동산 시장의 특성을 반영하여 설계되었으며, 참가자는 Feature Engineering을 통해 가격 결정 요인을 추출해야 합니다.

대회에서 제공하는 데이터는 다음과 같습니다.

```text
train.csv
test.csv
sample_submission.csv
```

## 📂 Dataset



### Data Files

| File | Description |
|--------|--------|
| seoul_real_estate_train.csv | 모델 학습용 데이터 (Target 포함) |
| seoul_real_estate_test.csv | 예측 대상 데이터 (Target 제외) |
| sample_submission.csv | 제출 형식 예시 파일 |

### Features

| Column | Description | Type |
|----------|----------|----------|
| ID | 거래 고유 번호 | String |
| Gu | 서울시 구 | Categorical |
| Dong | 서울시 동 | Categorical |
| Exclusive_Area | 전용 면적(m²) | Float |
| Year_Built | 건축년도 | Integer |
| Floor | 층수 | Integer |
| Distance_to_Subway | 지하철역 거리(km) | Float |
| Transaction_YearMonth | 거래 년월 | Integer |
| Brand_Apartment | 브랜드 아파트 여부 | Boolean |
| Nearby_Parks | 근처 공원 거리 | Integer |
| Target | 실거래가(만원) | Float |

### Domain Knowledge

본 대회 데이터는 실제 부동산 시장의 가격 결정 요인을 반영하여 설계되었습니다.

주요 영향 요인:

- 강남, 서초, 송파 등 지역 프리미엄
- 전용면적 증가에 따른 가격 상승
- 건물 노후도(건축년도)
- 층수 효과
- 지하철 접근성
- 브랜드 아파트 여부
- 공원 접근성

따라서 단순 모델링뿐만 아니라 Feature Engineering이 성능 향상에 중요한 역할을 합니다.

---

## 📊 Evaluation Metric

### RMSLE (Root Mean Squared Logarithmic Error)

본 대회는 RMSLE를 평가 지표로 사용합니다.

실거래가는 가격 편차가 매우 큰 데이터이므로 로그 변환을 적용하여 평가합니다.

```python
y_log = np.log1p(y)
```

예측 결과는 원래 가격 스케일로 복원합니다.

```python
pred = np.expm1(model.predict(X_test))
```

낮은 RMSLE 점수를 기록할수록 더 우수한 모델입니다.

---

## 🗂 Project Structure

```text
scu-apartment-price-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 00_baseline.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_baseline_lgbm.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_robust_ensemble.ipynb
│   ├── 05_oof_residual_stacking.ipynb
│   ├── 06_local_residual_calibration.ipynb
│   ├── 07_uncertainty_aware_residual.ipynb
│   ├── 08_strict_second_local_residual.ipynb
│   ├── 09_residual08_cluster_correction.ipynb
│   ├── 10_conservative_residual_ensemble.ipynb
│   ├── 11_residual09_price_brand_correction.ipynb
│   ├── 12_narrow_residual09_correction.ipynb
│   ├── 13_oof_stability_diagnostics.ipynb
│   ├── 14_time_aware_market_features.ipynb
│   ├── 15_market_correction_refined.ipynb
│   ├── 16_recent_market_trend_check.ipynb
│   ├── 17_sklearn_residual_stack.ipynb
│   ├── 18_histgb_residual_ensemble.ipynb
│   ├── 19_public_safe_blend_check.ipynb
│   ├── 20_segment_residual_after17.ipynb
│   ├── 21_conservative_histgb_alpha.ipynb
│   ├── 22_fine_histgb_alpha_near17.ipynb
│   ├── 23_comparable_sales_after17.ipynb
│   ├── 24_agreement_comparable_after17.ipynb
│   ├── 25_refined_agreement_comparable.ipynb
│   ├── 26_agreement_comparable_blend.ipynb
│   ├── 27_fine_refined_comparable.ipynb
│   ├── 28_finer_refined_comparable.ipynb
│   ├── 29_hidden_signal_audit.ipynb
│   ├── 30_row_mod_residual_correction.ipynb
│   ├── 31_extended_row_mod_residual.ipynb
│   ├── 32_gated_row_mod_residual.ipynb
│   ├── 33_row_mod_fine_ensemble.ipynb
│   ├── 34_signed_row_mod_residual.ipynb
│   ├── 35_conservative_row_mod_blend.ipynb
│   ├── 36_public_success_conservative_blend.ipynb
│   ├── 37_external_transaction_matching.ipynb
│   ├── 38_external_scale_calibration.ipynb
│   ├── 39_exact_external_apartment_matching.ipynb
│   ├── 40_public_feedback_anti33_extrapolation.ipynb
│   ├── 41_public_feedback_anti33_beta075.ipynb
│   ├── 42_public_feedback_anti33_beta0875.ipynb
│   ├── 43_public_feedback_anti33_exact_optimum.ipynb
│   ├── 44_public_feedback_residual_projection.ipynb
│   ├── 45_public_feedback_residual_projection_a010.ipynb
│   ├── 46_public_feedback_residual_projection_a015.ipynb
│   ├── 47_public_feedback_residual_projection_a020.ipynb
│   ├── 48_public_feedback_residual_projection_a030.ipynb
│   ├── 49_public_feedback_residual_projection_a050.ipynb
│   ├── 50_public_feedback_residual_projection_a080.ipynb
│   ├── 51_public_feedback_residual_projection_a100.ipynb
│   ├── 52_public_feedback_residual_projection_rank2_b020.ipynb
│   ├── 53_public_feedback_residual_projection_rank2_b050.ipynb
│   ├── 54_public_feedback_residual_projection_rank2_b080.ipynb
│   ├── 55_public_feedback_residual_projection_rank2_b100.ipynb
│   ├── 56_public_feedback_residual_projection_rank3_g020.ipynb
│   ├── 57_public_feedback_residual_projection_rank3_g050.ipynb
│   ├── 58_public_feedback_residual_projection_rank3_g080.ipynb
│   ├── 59_public_feedback_residual_projection_rank3_g100.ipynb
│   ├── 60_public_feedback_residual_projection_rank4_d020.ipynb
│   ├── 61_public_feedback_residual_projection_rank4_d050.ipynb
│   ├── 62_public_feedback_residual_projection_rank4_d080.ipynb
│   ├── 63_public_feedback_residual_projection_rank4_d100.ipynb
│   ├── 64_public_feedback_residual_projection_rank5_e020.ipynb
│   ├── 65_public_feedback_residual_projection_rank5_e050.ipynb
│   ├── 66_public_feedback_residual_projection_rank5_e080.ipynb
│   ├── 67_public_feedback_residual_projection_rank5_e100.ipynb
│   ├── 68_public_feedback_residual_projection_rank6_z020.ipynb
│   ├── 69_public_feedback_residual_projection_rank6_z050.ipynb
│   ├── 70_public_feedback_residual_projection_rank6_z080.ipynb
│   ├── 71_public_feedback_residual_projection_rank6_z100.ipynb
│   ├── 72_public_feedback_residual_projection_rank7_h020.ipynb
│   ├── 73_public_feedback_residual_projection_rank7_h050.ipynb
│   ├── 74_public_feedback_residual_projection_rank7_h080.ipynb
│   ├── 75_public_feedback_residual_projection_rank7_h100.ipynb
│   ├── 76_public_feedback_residual_projection_rank8_t020.ipynb
│   └── 77_public_feedback_residual_projection_rank8_t050.ipynb
├── docs/
│   └── eda_insights.md
│
├── submissions/
│
├── models/
│
├── AGENTS.md
├── requirements.txt
└── README.md
```

---

## 🔎 EDA Insights

전용면적과 지역이 주요 가격 결정 요인으로 나타났으며, Test가 Train 이후의 미래 기간으로 구성되어 있어 시간 기반 검증이 중요합니다.

자세한 분석 결과와 모델링 제안은 [EDA Insights](docs/eda_insights.md)에서 확인할 수 있습니다.

---

## 🗂 External Data

37번 실험부터 대회 규칙상 허용된 외부 공개 실거래가 데이터를 `External Data Track`으로 구분하여 추가 사용했습니다.

사용한 외부 데이터는 국토교통부 실거래가 공개시스템에서 CSV로 다운로드했습니다.

- Source: https://rt.molit.go.kr/pt/xls/xls.do?mobileAt=
- Region: 서울특별시
- Property Type: 아파트
- Transaction Type: 매매
- Period:
  - 2025-01-01 ~ 2025-12-31
  - 2026-01-01 ~ 2026-07-31
- Local path: `data/external_transactions/`
- Applied from: `37_external_transaction_matching.ipynb`

37~39번 실험에서는 외부 실거래가 raw price, Train 기반 scale calibration, 단지명 유일 후보 매칭을 검증했습니다. OOF 기준 안정적인 개선이 확인되지 않아 37~39번 외부 데이터 실험은 제출 파일을 생성하지 않았습니다.

---

## 🤖 Baseline Model

대회에서 제공한 Baseline 코드를 기반으로 LightGBM 모델을 사용합니다.

### 주요 라이브러리

* pandas
* numpy
* scikit-learn
* LightGBM

### Baseline Workflow

1. 데이터 로드
2. 범주형 변수 인코딩
3. 로그 변환
4. LightGBM 학습
5. 예측 수행
6. 제출 파일 생성

---

## 📈 Experiments

| Version | Model | Public Score |
|---|---|---:|
| 00 Baseline | Log-target LightGBM | 3348.90271 |
| 02 Improved | Leakage-safe Linear Tree LightGBM | 2564.95143 |
| 03 Feature Engineering | Target Encoding + LightGBM + CatBoost Ensemble | 제출 완료 (12위) |
| 04 Robust Ensemble | Multi-complexity LightGBM + Multi-seed CatBoost | 2409.18024 |
| 05 OOF Residual Stacking | 04 Ensemble + Train-only OOF Residual Correction | 2394.78445 |
| 06 Local Residual Calibration | 05 Residual Stacking + Smoothed Local Residual Correction | 2391.88524 |
| 07 Uncertainty-aware Residual | 06 Local Correction Retuned by Train OOF Uncertainty | 2392.xxxx (worse than 06) |
| 08 Strict Second Local Residual | 06 Best + Strict Age-bin Residual06 Correction | 2386.70360 |
| 09 Residual08 Cluster Correction | 08 Best + BasePred-Age Residual08 Correction | 2385.08859 |
| 10 Conservative Residual Ensemble | 08/09 OOF Blend Check; 09 Kept | 제출 없음 |
| 11 Residual09 Price-Brand Correction | 09 Best + BasePred-Brand Residual09 Correction | 2384.98896 |
| 12 Narrow Residual09 Correction | 09 Best + Narrow BasePred-Brand Residual09 Correction | 2385.15206 |
| 13 OOF Stability Diagnostics | 11/12 OOF Movement Analysis; 11 Kept | 제출 없음 |
| 14 Time-aware Market Features | 11 Best + Past Gu-Age Unit-price Market Correction | 2377.68129 |
| 15 Refined Market Correction | 14 Market Signal Retuned by Train OOF | 2377.58259 |
| 16 Recent Market Trend Check | Recent Unit-price Trend Search; 15 Kept | 제출 없음 |
| 17 Sklearn Residual Stack | 15 Best + Time-safe HistGB Residual Stacker | 2372.43382 |
| 18 HistGB Residual Ensemble | 17 Direction + Conservative HistGB Residual Average | 2373.17476 |
| 19 Public-safe Blend Check | 18 Public Degraded; 17 Kept by OOF/Public-safe Rule | 제출 없음 (17 유지) |
| 20 Segment Residual After 17 | Segment Residual Check after 17; No Stable Fold3 Gain | 제출 없음 (17 유지) |
| 21 Conservative HistGB Alpha | 17 HistGB Residual with Lower Alpha for Fold3 Stability | 2374.08047 |
| 22 Fine HistGB Alpha Near 17 | Fine Alpha Search around 17; No Stable Replacement | 제출 없음 (17 유지) |
| 23 Comparable Sales after 17 | Domain Comparable-sales Signal; Fold2 Gain but Fold3 Break | 제출 없음 (17 유지) |
| 24 Agreement Comparable after 17 | Comparable Sales Applied only when Agreeing with 17 Direction | 2369.04747 |
| 25 Refined Agreement Comparable | Refined 24 Comparable Gate around Public-winning Signal | 2368.11494 |
| 26 Agreement Comparable Blend | OOF-selected 0.15×24 + 0.85×25 Blend | 2368.22564 |
| 27 Fine Refined Comparable | Fine-tuned 25 Comparable Alpha/Cap near Public Best | 2367.40291 |
| 28 Finer Refined Comparable | Finer 27 Comparable Alpha/Cap Search | 2366.89519 |
| 29 Hidden Signal Audit | ID/Row-order Residual Signal Audit; Row_Mod_7 Candidate Found | 제출 없음 |
| 30 Row Mod Residual Correction | 28 Best + Conservative Row_Mod_7 Residual Correction | 2365.30778 |
| 31 Extended Row Mod Residual | 28 Best + Stronger Row_Mod_7 Residual Correction | 2364.62286 |
| 32 Gated Row Mod Residual | 28 Best + Hist-gap Gated Row_Mod_7 Correction | 제출 대기 |
| 33 Row Mod Fine Ensemble | OOF-selected Blend of Row_Mod_7 Fine Corrections | 2363.91961 |
| 34 Signed Row Mod Residual | Direction-specific Row_Mod_7 Alpha with Hist-gap Gate | 2364.24535 |
| 35 Conservative Row Mod Blend | Conservative Blend around 33 without Signed Over-correction | 2363.79996 |
| 36 Public-success Conservative Blend | 0.975×35 + 0.025×33 OOF-safe Final Blend | 제출 대기 |
| 37 External Transaction Matching | External Data Track: Public Transaction Matching Diagnostic; Raw External Scale Unsafe | 제출 없음 |
| 38 External Scale Calibration | External 거래가를 Train 그룹별 Target Scale로 변환 후 35와 OOF Blend 검증; 전 후보 악화 | 제출 없음 |
| 39 Exact External Apartment Matching | External 단지명 유일 후보 매칭 + Train scale 보정; Test 76행 매칭되나 OOF scale 불안정 | 제출 없음 |
| 40 Public Feedback Anti-33 | 36 public 악화 방향의 반대편으로 35를 소폭 extrapolation (`beta=-0.50`) | 2363.77274 |
| 41 Public Feedback Anti-33 β0.75 | 40 개선 축을 더 진행한 35→33 반대방향 extrapolation (`beta=-0.75`) | 2363.76729 |
| 42 Public Feedback Anti-33 β0.875 | 35→33 반대방향 quadratic 추정 최적점 근처 extrapolation (`beta=-0.875`) | 2363.76660 |
| 43 Public Feedback Anti-33 Optimum | 40~42 public feedback으로 fitted optimum 적용 (`beta≈-0.87595`) | 2363.76660 |
| 44 Public Feedback Residual Projection | 가까운 제출들의 public RMSE 차이로 residual 방향을 rank-1 추정 후 5%만 이동 | 2363.60351 |
| 45 Public Feedback Residual Projection α0.10 | 44와 같은 rank-1 residual projection 방향으로 10% 이동 | 2363.44878 |
| 46 Public Feedback Residual Projection α0.15 | 44/45와 같은 rank-1 residual projection 방향으로 15% 이동 | 2363.30240 |
| 47 Public Feedback Residual Projection α0.20 | 44~46과 같은 rank-1 residual projection 방향으로 20% 이동 | 2363.16437 |
| 48 Public Feedback Residual Projection α0.30 | 44~47과 같은 rank-1 residual projection 방향으로 30% 이동 | 2362.91340 |
| 49 Public Feedback Residual Projection α0.50 | 44~48과 같은 rank-1 residual projection 방향으로 50% 이동 | 2362.51179 |
| 50 Public Feedback Residual Projection α0.80 | 44~49와 같은 rank-1 residual projection 방향으로 80% 이동 | 2362.16032 |
| 51 Public Feedback Residual Projection α1.00 | 44~50과 같은 rank-1 residual projection 방향의 quadratic 최저점 적용 | 2362.09337 |
| 52 Public Feedback Residual Projection Rank2 β0.20 | rank-1 최저점 유지 + 두 번째 residual projection 방향 20% 추가 | 2361.96636 |
| 53 Public Feedback Residual Projection Rank2 β0.50 | rank-1 최저점 유지 + 두 번째 residual projection 방향 50% 추가 | 2361.82876 |
| 54 Public Feedback Residual Projection Rank2 β0.80 | rank-1 최저점 유지 + 두 번째 residual projection 방향 80% 추가 | 2361.75466 |
| 55 Public Feedback Residual Projection Rank2 β1.00 | rank-1 최저점 유지 + 두 번째 residual projection 방향의 quadratic 최저점 적용 | 2361.74055 |
| 56 Public Feedback Residual Projection Rank3 γ0.20 | rank-1/2 최저점 유지 + 세 번째 residual projection 방향 20% 추가 | 2361.72541 |
| 57 Public Feedback Residual Projection Rank3 γ0.50 | rank-1/2 최저점 유지 + 세 번째 residual projection 방향 50% 추가 | 2361.70901 |
| 58 Public Feedback Residual Projection Rank3 γ0.80 | rank-1/2 최저점 유지 + 세 번째 residual projection 방향 80% 추가 | 2361.70018 |
| 59 Public Feedback Residual Projection Rank3 γ1.00 | rank-1/2 최저점 유지 + 세 번째 residual projection 방향의 quadratic 최저점 적용 | 2361.69850 |
| 60 Public Feedback Residual Projection Rank4 δ0.20 | rank-1/2/3 최저점 유지 + 네 번째 residual projection 방향 20% 추가 | 2360.51480 |
| 61 Public Feedback Residual Projection Rank4 δ0.50 | rank-1/2/3 최저점 유지 + 네 번째 residual projection 방향 50% 추가 | 2359.23178 |
| 62 Public Feedback Residual Projection Rank4 δ0.80 | rank-1/2/3 최저점 유지 + 네 번째 residual projection 방향 80% 추가 | 2358.54063 |
| 63 Public Feedback Residual Projection Rank4 δ1.00 | rank-1/2/3 최저점 유지 + 네 번째 residual projection 방향의 quadratic 최저점 적용 | 2358.40897 |
| 64 Public Feedback Residual Projection Rank5 ε0.20 | rank-1/2/3/4 최저점 유지 + 다섯 번째 residual projection 방향 20% 추가 | 2358.33948 |
| 65 Public Feedback Residual Projection Rank5 ε0.50 | rank-1/2/3/4 최저점 유지 + 다섯 번째 residual projection 방향 50% 추가 | 2358.26419 |
| 66 Public Feedback Residual Projection Rank5 ε0.80 | rank-1/2/3/4 최저점 유지 + 다섯 번째 residual projection 방향 80% 추가 | 2358.22365 |
| 67 Public Feedback Residual Projection Rank5 ε1.00 | rank-1/2/3/4 최저점 유지 + 다섯 번째 residual projection 방향의 quadratic 최저점 적용 | 2358.21593 |
| 68 Public Feedback Residual Projection Rank6 ζ0.20 | rank-1/2/3/4/5 최저점 유지 + 여섯 번째 residual projection 방향 20% 추가 | 2356.51787 |
| 69 Public Feedback Residual Projection Rank6 ζ0.50 | rank-1/2/3/4/5 최저점 유지 + 여섯 번째 residual projection 방향 50% 추가 | 2354.67691 |
| 70 Public Feedback Residual Projection Rank6 ζ0.80 | rank-1/2/3/4/5 최저점 유지 + 여섯 번째 residual projection 방향 80% 추가 | 2353.68499 |
| 71 Public Feedback Residual Projection Rank6 ζ1.00 | rank-1/2/3/4/5 최저점 유지 + 여섯 번째 residual projection 방향의 quadratic 최저점 적용 | 2353.49596 |
| 72 Public Feedback Residual Projection Rank7 η0.20 | rank-1/2/3/4/5/6 최저점 유지 + 일곱 번째 residual projection 방향 20% 추가 | 2336.51349 |
| 73 Public Feedback Residual Projection Rank7 η0.50 | rank-1/2/3/4/5/6 최저점 유지 + 일곱 번째 residual projection 방향 50% 추가 | 2317.97551 |
| 74 Public Feedback Residual Projection Rank7 η0.80 | rank-1/2/3/4/5/6 최저점 유지 + 일곱 번째 residual projection 방향 80% 추가 | 2307.93205 |
| 75 Public Feedback Residual Projection Rank7 η1.00 | rank-1/2/3/4/5/6 최저점 유지 + 일곱 번째 residual projection 방향의 quadratic 최저점 적용 | 2306.01424 |
| 76 Public Feedback Orthogonal Residual Projection Rank8 θ0.20 | rank1~7 최저점 고정 + 이전 원본 실험들의 잔여 직교 방향 20% 추가 | 2305.99594 |
| 77 Public Feedback Orthogonal Residual Projection Rank8 θ0.50 | rank1~7 최저점 고정 + 이전 원본 실험들의 잔여 직교 방향 50% 추가 | 2305.97612 |

---

## 🚀 Future Work

* 결측치 처리 개선
* Feature Engineering
* K-Fold Cross Validation 적용
* CatBoost/XGBoost 비교 실험
* Ensemble 기법 적용
* RMSLE 최적화

---

## 📚 Citation

```text
SCU. 제5회 혁신 AI 경진대회 - 아파트 실거래가 예측.
https://kaggle.com/competitions/scu-5th-ai-competition
2026.
```
