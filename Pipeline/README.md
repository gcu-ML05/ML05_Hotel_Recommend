# 모델링 파이프라인

전처리된 데이터(`data/processing/`)로 제안서의 네 가지 필터링 방법을 학습하고 평가합니다.

| 제안서            | 구현                                                                    |
| -------------- | --------------------------------------------------------------------- |
| ① 메모리 기반 CF    | Item-kNN (`models.ItemKNN`)                                           |
| ② 모델 기반 CF     | 편향 포함 행렬분해, L2 정규화, SGD (`models.BiasedMF`)                           |
| ③ 콘텐츠 기반       | 사용자 aspect 중요도 × 호텔 aspect 평균 + 리뷰 TF-IDF 유사도 (`models.ContentBased`) |
| ④ 하이브리드        | λ · MF + (1 − λ) · 콘텐츠 기반                                             |
| 선형회귀 · K-means | 사용자별 aspect 중요도 추정, 여행자 페르소나                                          |

- **HotelRec (TripAdvisor)**: 모델 설계·학습·평가 (사용자별 최신 리뷰로 test)
- **515K Hotel Reviews (Booking.com, Kaggle)**: 신규 사용자 일반화 테스트 (사용자 ID가 없어 CF 직접 평가는 불가)

## 1. 전체 흐름

```text
data/processing/hotelrec/ ─┬─ 01_split.py ───────────── 학습/검증/테스트 분할
                           ├─ 02_user_aspect_weights.py 사용자 aspect 중요도 + 페르소나
                           ├─ 03_hotel_features.py ──── 호텔 aspect 평균 + TF-IDF
                           └─ 04_train_evaluate.py ──── 모델 학습·평가 (models.py)
data/processing/kaggle/  ──┬─ 05_kaggle_generalization.py  리뷰 텍스트로 여행 유형별 선호 검증
                           └─ 06_kaggle_hotel_matching.py  같은 호텔에 대한 예측 vs 실제 점수
```

각 단계의 중간 산출물은 `data/pipeline/`(Git 제외)에 저장되고 다음 단계가 읽습니다. 결과 표(csv)와 그림(png)은 `Pipeline/outputs/`에 저장됩니다.

## 2. 준비

**라이브러리**: 저장소 루트의 `requirements`로 설치합니다 (파이프라인은 numpy, pandas, scikit-learn, matplotlib을 쓰고, scipy는 scikit-learn과 함께 설치됩니다).

**데이터**: 팀 Drive의 전처리 결과를 아래처럼 둡니다.

```text
data/processing/hotelrec/  hotelrec_train.csv, hotelrec_partial.csv, hotel_map.csv, user_map.csv
data/processing/kaggle/    kaggle_test.csv, hotel_meta_kaggle.csv
```

> `hotel_map.csv`는 **46,747행** 버전이어야 합니다. 처음 공유된 162,023행 버전은 리뷰 데이터와 `hotel_id`가 맞지 않습니다. 이 경우 04의 같은 도시 평가가 자동으로 건너뛰어지고, 06의 호텔 매칭 결과도 틀립니다.

## 3. 실행

`Pipeline/` 폴더에서 순서대로 실행합니다.

```bash
cd Pipeline
python 01_split.py                  # 약 1분
python 02_user_aspect_weights.py    # 약 2분
python 03_hotel_features.py         # 약 5분
python 04_train_evaluate.py         # 약 15분
python 05_kaggle_generalization.py  # 약 30초
python 06_kaggle_hotel_matching.py  # 약 1분 (04 이후)
python 04a_tune_itemknn.py          # 선택: Item-kNN 설정 선택 근거 (약 30분)
```

`04_train_evaluate.py`는 옵션으로 하이퍼파라미터를 바꿀 수 있습니다 (`python 04_train_evaluate.py -h`).

## 4. 파일 구성

| 파일 | 역할 | 주요 출력 |
| --- | --- | --- |
| `common.py` | 경로, 데이터 로드, RMSE/MAE, HR@k/NDCG@k, Top-N 후보 생성 | – |
| `models.py` | 모든 추천 모델 (같은 `fit` / `predict` 인터페이스) | – |
| `01_split.py` | 사용자별 시간 순 분할 | `01_split_summary.csv` |
| `02_user_aspect_weights.py` | 전역/사용자별 aspect 가중치, 페르소나 | `02_*.csv`, `figures/02_*.png` |
| `03_hotel_features.py` | 호텔 aspect 평균, 호텔 TF-IDF | `03_hotel_top_terms.csv` |
| `04_train_evaluate.py` | 모델 학습, 평점 예측·Top-N 평가, 하이브리드 λ 선택 | `04_*.csv`, `figures/04_*.png` |
| `04a_tune_itemknn.py` | Item-kNN 설정 비교 (val) | `04a_itemknn_tuning.csv` |
| `05_kaggle_generalization.py` | Kaggle 리뷰의 aspect 언급 분석 | `05_*.csv`, `figures/05_*.png` |
| `06_kaggle_hotel_matching.py` | HotelRec·Kaggle 공통 호텔에서 예측 vs 실제 점수 | `06_*.csv`, `figures/06_*.png` |

## 5. 단계별 방법

### 01 분할
train(세부 평점 완전)과 partial을 합친 전체 리뷰에서 **사용자마다 가장 최근 리뷰는 test, 그 직전 리뷰는 val, 나머지는 train**으로 나눕니다. 미래 리뷰로 과거를 맞히는 시간 누수가 없습니다. 날짜가 월 단위라 같은 달의 리뷰는 시드 42 난수로 순서를 정합니다.

### 02 사용자 aspect 중요도와 페르소나
- 전역 선형회귀: `rating ≈ w0 + Σ w_k · aspect_k` (세부 평점이 모두 있는 train 리뷰)
- 사용자별 Ridge: 사용자당 완전 리뷰가 평균 3.7건뿐이라, 계수를 0이 아닌 **전역 가중치 쪽으로 당기는** 정규화를 씁니다. 강도 λ는 val로 고릅니다.
- 페르소나: 가중치를 중요도 비율로 바꿔 K-means로 군집화합니다 (K는 실루엣 점수로 선택).

### 03 호텔 특징
- 호텔 aspect 평균: train 리뷰의 세부 평점을 aspect별로 평균하고, 리뷰가 적은 호텔은 전체 평균 쪽으로 축소합니다.
- 호텔 TF-IDF: train 리뷰의 제목과 본문을 호텔별로 모아 TF-IDF를 계산하고, 호텔마다 상위 100개 단어만 남깁니다.

### 04 모델과 평가

| 모델 | 방법 |
| --- | --- |
| Popularity / PersonaPopularity | 리뷰 수가 많은 호텔 / 같은 페르소나가 높게 평가한 호텔 (기준선) |
| Baseline | μ + b_u + b_i |
| ItemKNN | 평점 예측: 기준 예측(μ+b_u+b_i) + 이웃 호텔 잔차의 가중 평균 (KNNBaseline)<br>Top-N: 사용자가 다녀간 호텔들과의 공동 방문 유사도 합 |
| MF | μ + b_u + b_i + p_u·q_i, MSE + L2, 미니배치 SGD, val 기준 조기 종료 |
| Content | aspect 효용 (사용자 가중치 × 호텔 aspect 평균) + γ × TF-IDF 코사인 유사도 |
| Hybrid | λ · MF + (1 − λ) · Content, λ는 val NDCG@10으로 선택 |

평가 지표:
- **평점 예측**: 모든 사용자의 test 리뷰에 대한 RMSE, MAE
- **Top-N**: 정답 호텔(test 리뷰의 호텔) 1개와 사용자가 평가하지 않은 후보를 섞어 HR@k, NDCG@k를 계산합니다.
  - 무작위 후보: 전 세계 호텔 99개 (test 사용자 2만 명 표본)
  - 같은 도시 후보: 정답과 같은 도시의 호텔 49개. "여행지를 정한 뒤 호텔 고르기"에 해당하며, 호텔이 50개 이상인 도시의 사용자만 평가됩니다.

### 05 Kaggle 텍스트 기반 검증
Kaggle 리뷰 1건을 신규 사용자 1명으로 봅니다. aspect별 키워드 사전(`LEXICON`)으로 긍정·부정 리뷰에서 언급된 aspect를 찾습니다. 이를 바탕으로 (1) 여행 유형별 언급 차이, (2) HotelRec 페르소나와의 대응, (3) HotelRec 가중치로 매긴 호텔 점수와 실제 점수의 상관을 봅니다.

### 06 Kaggle 호텔 매칭
같은 도시 안에서 호텔명을 정규화해 두 데이터의 같은 호텔을 찾습니다. 정확히 일치하거나, 문자열 유사도가 0.9 이상인 경우만 인정합니다. 그다음 HotelRec 모델의 **신규 사용자 예측**(사용자 항 0)이 Booking.com 실제 점수와 얼마나 맞는지 평가합니다.

## 6. 주요 결과

시드 42 기준입니다. 전체 표는 `outputs/`에 있습니다.

**평점 예측 (test)**

| 모델 | RMSE | MAE |
| --- | --- | --- |
| GlobalMean | 1.004 | 0.784 |
| Baseline | 0.909 | 0.700 |
| ItemKNN | 0.909 | 0.700 |
| **MF** | **0.907** | **0.693** |
| Content (aspect + text) | 0.952 | 0.745 |

**Top-N (test HR@10 / NDCG@10)**

| 모델 | 무작위 후보 (1+99) | 같은 도시 후보 (1+49) |
| --- | --- | --- |
| **ItemKNN** | **0.349 / 0.211** | 0.392 / 0.191 |
| Popularity | 0.328 / 0.176 | **0.479 / 0.255** |
| Baseline | 0.146 / 0.073 | 0.312 / 0.152 |
| Content = Hybrid (λ=0) | 0.135 / 0.065 | 0.296 / 0.143 |
| MF | 0.117 / 0.053 | 0.285 / 0.131 |
| 무작위로 순위를 매길 때 | 0.100 | 0.200 |

**Kaggle 공통 호텔 505개 (리뷰 203,215건): 신규 사용자 예측 vs 실제 점수**

| 모델 | 리뷰 RMSE | 호텔 평균 점수와 상관 |
| --- | --- | --- |
| GlobalMean | 0.812 | – |
| MF (μ + b_i) | 0.784 | **0.743** |
| Content (aspect) | 0.788 | 0.674 |

**해석**
- 평점 예측은 MF가 가장 좋지만 Baseline과 차이가 작습니다. 사용자당 리뷰가 평균 7건이라 개인화로 얻을 정보가 적습니다.
- Top-N에서는 ItemKNN(공동 방문)과 인기도가 가장 강합니다. 평점을 예측하는 모델(MF, Content)은 "어느 호텔에 갈지"를 맞히는 데 약합니다. 그래서 하이브리드도 val에서 λ=0(콘텐츠 기반만)이 선택됐습니다.
- ItemKNN은 이력이 많은 사용자일수록 좋아지고(HR@10: 리뷰 3~4건 0.299 → 20건 이상 0.472), 콘텐츠 기반은 이력이 적은 사용자에서 상대적으로 낫습니다.
- **사용자별 aspect 가중치는 전역 가중치보다 낫지 않았습니다.** val에서 개인화를 약하게 할수록 오차가 줄었고, 그 결과 페르소나 간 차이도 거의 없습니다(중요도 비율 차이 약 0.003).
- 반면 **aspect 가중치 자체는 플랫폼을 넘어 일반화됩니다.** HotelRec 전역 가중치로 매긴 Booking.com 호텔 점수는 동일 가중보다 실제 점수와 상관이 높습니다(0.274 → 0.301).
- Booking.com 리뷰에서도 여행 유형별 선호 차이가 보입니다. 가족 여행객은 location을, 출장객은 rooms·sleep_quality·value를 더 언급합니다.
- **HotelRec에서 배운 호텔 점수는 Booking.com 호텔 평균 점수와 상관 0.74**로, 다른 플랫폼의 신규 사용자에게도 호텔 수준의 예측은 통합니다.

## 7. 한계와 남은 과제

- **하이브리드 결합 방식**: 지금 결합(MF + Content)은 순위에서 효과가 없습니다. CF 쪽을 ItemKNN으로 바꾸거나 인기도를 함께 넣는 방안을 검토해야 합니다.
- **페르소나**: 회귀 계수로는 사용자 유형이 구별되지 않습니다. 완전 리뷰가 많은 사용자만 분석하거나, 다른 선호 표현으로 군집화하거나, Kaggle 여행 유형을 활용하는 방안이 있습니다.
- **평가 범위**: 같은 도시 평가는 대도시 사용자(표본의 25%)만 포함합니다. Kaggle 매칭은 6개 유럽 도시 505개 호텔에 한정됩니다.
- **수작업 규칙**: 05의 aspect 키워드 사전과 06의 호텔명 매칭 규칙은 직접 정한 것입니다. 매칭 목록은 `outputs/06_matched_hotels.csv`에서 확인할 수 있습니다.

## 8. 참고 문헌

- Antognini & Faltings (2020) HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset. LREC.
- Sarwar et al. (2001) Item-based collaborative filtering recommendation algorithms. WWW.
- Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS.
- Koren, Bell, Volinsky (2009) Matrix factorization techniques for recommender systems. IEEE Computer.
- Koren (2010) Factor in the neighbors: Scalable and accurate collaborative filtering. ACM TKDD.
