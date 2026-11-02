# 모델링 파이프라인

전처리된 데이터(`data/processing/`)로 제안서의 네 가지 필터링 방법을 학습하고 평가합니다. 사용자의 aspect 선호와 여행자 페르소나를 추정하고, 다른 플랫폼(Booking.com)의 신규 사용자에게도 통하는지 검증합니다.

| 제안서 | 구현 |
| --- | --- |
| ① 메모리 기반 CF | Item-kNN (`models.ItemKNN`) |
| ② 모델 기반 CF | 편향 포함 행렬분해, L2 정규화, SGD (`models.BiasedMF`) |
| ③ 콘텐츠 기반 | 사용자 aspect 중요도 × 호텔 aspect 평균 + 리뷰 TF-IDF 유사도 (`models.ContentBased`) |
| ④ 하이브리드 | λ · MF + (1 − λ) · 콘텐츠 기반 |
| 선형회귀 · K-means | 사용자 aspect 중요도 추정, 여행자 페르소나 |

- **HotelRec (TripAdvisor)**: 모델 설계·학습·평가 (사용자별 최신 리뷰로 test)
- **515K Hotel Reviews (Booking.com, Kaggle)**: 신규 사용자 일반화 테스트 (사용자 ID가 없어 CF 직접 평가는 불가)

## 1. 전체 흐름

```text
data/processing/hotelrec/
  │
  ├─ 01_split ─────────────────── 학습/검증/테스트 분할
  ├─ 02_user_aspect_weights ───── 회귀 기반 aspect 가중치, 페르소나
  ├─ 03_hotel_features ────────── 호텔 aspect 평균, TF-IDF
  ├─ 04_persona_deviation ─────── 편차 기반 사용자 선호(관대함·까다로움), 페르소나
  ├─ 05_persona_tuning ────────── 04의 설정 선택과 test 평가
  ├─ 06_train_evaluate ────────── 추천 모델 학습·평가 (models.py)
  └─ 07_tune_itemknn (선택) ───── Item-kNN 설정 선택 근거
data/processing/kaggle/
  ├─ 08_kaggle_generalization ─── 리뷰 aspect 언급으로 여행 유형별 선호 검증
  ├─ 09_lexicon_audit ─────────── 키워드 사전 점검
  ├─ 10_persona_lexicon_compare ─ 페르소나·키워드 사전 비교
  └─ 11_kaggle_hotel_matching ─── 공통 호텔에서 예측 vs 실제 점수
```

번호 순서대로 실행하면 각 단계가 필요한 입력을 앞 단계에서 받습니다. 중간 산출물은 `data/pipeline/`(Git 제외), 결과 표(csv)와 그림(png)은 `Pipeline/outputs/`에 저장되며, 결과 파일 이름 앞의 번호는 그것을 만든 스크립트 번호입니다.

## 2. 준비

**라이브러리**: 저장소 루트의 `requirements`로 설치합니다 (파이프라인은 numpy, pandas, scikit-learn, matplotlib을 쓰고, scipy는 scikit-learn과 함께 설치됩니다).

**데이터**: 팀 Drive의 전처리 결과를 아래처럼 둡니다.

```text
data/processing/hotelrec/  hotelrec_train.csv, hotelrec_partial.csv, hotel_map.csv, user_map.csv
data/processing/kaggle/    kaggle_test.csv, hotel_meta_kaggle.csv
```

> `hotel_map.csv`는 **46,747행** 버전이어야 합니다. 처음 공유된 162,023행 버전은 리뷰 데이터와 `hotel_id`가 맞지 않습니다. 이 경우 06의 같은 도시 평가가 자동으로 건너뛰어지고, 11의 호텔 매칭 결과도 틀립니다.

## 3. 실행

`Pipeline/` 폴더에서 번호 순서대로 실행합니다. 전체 약 25분입니다.

```bash
cd Pipeline
python 01_split.py                    # 약 0.5분
python 02_user_aspect_weights.py      # 약 2분
python 03_hotel_features.py           # 약 4분
python 04_persona_deviation.py        # 약 1분
python 05_persona_tuning.py           # 약 0.5분
python 06_train_evaluate.py           # 약 6분
python 07_tune_itemknn.py             # 선택, 약 9분
python 08_kaggle_generalization.py    # 약 0.5분
python 09_lexicon_audit.py            # 약 1분
python 10_persona_lexicon_compare.py  # 약 0.5분
python 11_kaggle_hotel_matching.py    # 약 0.1분
```

`06_train_evaluate.py`와 `04_persona_deviation.py`는 옵션으로 설정을 바꿀 수 있습니다 (`python 06_train_evaluate.py -h`).

## 4. 파일 구성

| 파일 | 역할 | 입력 (앞 단계) | 주요 출력 |
| --- | --- | --- | --- |
| `common.py` | 경로, 데이터 로드, RMSE/MAE, HR@k/NDCG@k, Top-N 후보 생성 | – | – |
| `models.py` | 추천 모델 (같은 `fit` / `predict` 인터페이스) | – | – |
| `01_split.py` | 사용자별 시간 순 분할 | 전처리 결과 | `01_split_summary.csv` |
| `02_user_aspect_weights.py` | 회귀 기반 aspect 가중치, 페르소나 | 01 | `02_*.csv`, `figures/02_*.png` |
| `03_hotel_features.py` | 호텔 aspect 평균, 호텔 TF-IDF | 01 | `03_hotel_top_terms.csv` |
| `04_persona_deviation.py` | 편차 기반 관대함·까다로움, 페르소나 | 01, 02, 03 | `04_*.csv`, `figures/04_*.png` |
| `05_persona_tuning.py` | 04의 축소 강도·관대함 계수·K 선택, test 평가 | 01~04 | `05_*.csv`, `figures/05_*.png` |
| `06_train_evaluate.py` | 추천 모델 학습, 평점 예측·Top-N 평가, 하이브리드 λ 선택 | 01, 02, 03 | `06_*.csv`, `figures/06_*.png` |
| `07_tune_itemknn.py` | Item-kNN 설정 비교 (선택) | 01 | `07_itemknn_tuning.csv` |
| `08_kaggle_generalization.py` | Kaggle 리뷰의 aspect 언급 분석, 키워드 사전(`LEXICON`) | 02 | `08_*.csv`, `figures/08_*.png` |
| `09_lexicon_audit.py` | 키워드별 빈도·동시 언급·민감도 점검 | 08의 사전 | `09_*.csv` |
| `10_persona_lexicon_compare.py` | 페르소나 3종 × 사전 2종 비교, 사전 강건성 | 02, 04, 05, 08 | `10_*.csv` |
| `11_kaggle_hotel_matching.py` | HotelRec·Kaggle 공통 호텔에서 예측 vs 실제 점수 | 02, 03, 06 | `11_*.csv`, `figures/11_*.png` |

## 5. 단계별 방법

### 01 분할
train(세부 평점 완전)과 partial을 합친 전체 리뷰에서 **사용자마다 가장 최근 리뷰는 test, 그 직전 리뷰는 val, 나머지는 train**으로 나눕니다. 미래 리뷰로 과거를 맞히는 시간 누수가 없습니다. 날짜가 월 단위라 같은 달의 리뷰는 시드 42 난수로 순서를 정합니다.

### 02 회귀 기반 aspect 가중치
- 전역 선형회귀: `rating ≈ w0 + Σ w_k · aspect_k` (세부 평점이 모두 있는 train 리뷰)
- 사용자별 Ridge: 사용자당 완전 리뷰가 평균 3.7건뿐이라, 계수를 0이 아닌 **전역 가중치 쪽으로 당기는** 정규화를 씁니다. 강도 λ는 val로 고릅니다.
- 페르소나: 가중치를 중요도 비율로 바꿔 K-means로 군집화합니다 (K는 실루엣 점수로 선택).

### 03 호텔 특징
- 호텔 aspect 평균: train 리뷰의 세부 평점을 aspect별로 평균하고, 리뷰가 적은 호텔은 전체 평균 쪽으로 축소합니다.
- 호텔 TF-IDF: train 리뷰의 제목과 본문을 호텔별로 모아 TF-IDF를 계산하고, 호텔마다 상위 100개 단어만 남깁니다.

### 04·05 편차 기반 사용자 선호와 페르소나
02의 회귀 계수는 리뷰가 적어 사용자마다 구별되지 않았습니다. 그래서 계수를 추정하는 대신, 사용자가 준 aspect 점수가 **그 호텔의 평균 aspect 점수보다 얼마나 높거나 낮은지**를 평균합니다.
- **관대함** L_u: 모든 aspect에서 호텔 평균보다 후하게(+) 또는 박하게(−) 주는 정도
- **까다로움**: 관대함을 뺀 aspect별 상대 편차의 부호를 바꾼 값. 클수록 그 aspect를 다른 사람보다 박하게 평가합니다. 리뷰가 적은 aspect는 0 쪽으로 축소(M)합니다.
- 페르소나: 까다로움 벡터를 K-means로 군집화합니다 (예: "서비스 깐깐형").
- 검증: val/test 리뷰의 aspect 점수를 `호텔 평균 + 관대함 (+ 페르소나 또는 개인 까다로움)`으로 예측한 RMSE. 05는 축소 강도 M, aspect별 관대함 계수 β, K를 val로 고르고 test로 최종 보고합니다.

### 06 추천 모델과 평가

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

07은 Item-kNN의 기준 예측 방식, 이웃 수 k, 축소 강도를 val로 비교해 06의 기본값(k=2000, shrink=100, damp=0.03)을 고른 근거를 남깁니다.

### 08~10 Kaggle 텍스트 검증
Kaggle 리뷰 1건을 신규 사용자 1명으로 봅니다. aspect별 키워드 사전(`08`의 `LEXICON`)으로 긍정·부정 리뷰에서 언급된 aspect를 찾습니다.
- 08: 여행 유형별 aspect 언급 차이, HotelRec 페르소나와의 대응, HotelRec 가중치로 매긴 호텔 점수와 실제 점수의 상관
- 09: 사전 점검. 키워드별 빈도, 같은 리뷰에서 두 aspect가 함께 잡히는 비율, 키워드를 하나씩 뺐을 때 결과가 바뀌는 정도
- 10: 09의 점검을 반영한 수정 사전과 원본 사전, 회귀 기반·편차 기반 페르소나를 같은 지표로 비교하고, 사전을 바꿔도 결론이 유지되는지 확인

### 11 Kaggle 호텔 매칭
같은 도시 안에서 호텔명을 정규화해 두 데이터의 같은 호텔을 찾습니다. 정확히 일치하거나, 문자열 유사도가 0.9 이상인 경우만 인정합니다. 그다음 HotelRec 모델의 **신규 사용자 예측**(사용자 항 0)이 Booking.com 실제 점수와 얼마나 맞는지 평가합니다.

## 6. 주요 결과

시드 42 기준입니다. 전체 표는 `outputs/`에 있습니다.

**평점 예측 (06, test)**

| 모델 | RMSE | MAE |
| --- | --- | --- |
| GlobalMean | 1.004 | 0.784 |
| Baseline | 0.909 | 0.700 |
| ItemKNN | 0.909 | 0.700 |
| **MF** | **0.907** | **0.693** |
| Content (aspect + text) | 0.952 | 0.745 |

**Top-N (06, test HR@10 / NDCG@10)**

| 모델 | 무작위 후보 (1+99) | 같은 도시 후보 (1+49) |
| --- | --- | --- |
| **ItemKNN** | **0.349 / 0.211** | 0.392 / 0.191 |
| Popularity | 0.328 / 0.176 | **0.479 / 0.255** |
| Baseline | 0.146 / 0.073 | 0.312 / 0.152 |
| Content = Hybrid (λ=0) | 0.135 / 0.065 | 0.296 / 0.143 |
| MF | 0.117 / 0.053 | 0.285 / 0.131 |
| 무작위로 순위를 매길 때 | 0.100 | 0.200 |

**사용자 aspect 선호 (05, test 세부 평점 예측 RMSE)**

| 예측 | RMSE |
| --- | --- |
| 호텔 평균만 | 0.905 |
| + 관대함 | 0.890 |
| + aspect별 관대함 (β) | 0.878 |
| + 페르소나 (K=2) | 0.878 |
| **+ 개인 까다로움 (M=30)** | **0.877** |

**Kaggle 공통 호텔 505개 (11, 리뷰 203,215건): 신규 사용자 예측 vs 실제 점수**

| 모델 | 리뷰 RMSE | 호텔 평균 점수와 상관 |
| --- | --- | --- |
| GlobalMean | 0.812 | – |
| MF (μ + b_i) | 0.784 | **0.743** |
| Content (aspect) | 0.788 | 0.674 |

**해석**
- 평점 예측은 MF가 가장 좋지만 Baseline과 차이가 작습니다. 사용자당 리뷰가 평균 7건이라 개인화로 얻을 정보가 적습니다.
- Top-N에서는 ItemKNN(공동 방문)과 인기도가 가장 강합니다. 평점을 예측하는 모델(MF, Content)은 "어느 호텔에 갈지"를 맞히는 데 약해, 하이브리드도 val에서 λ=0(콘텐츠 기반만)이 선택됐습니다.
- ItemKNN은 이력이 많은 사용자일수록 좋아지고(HR@10: 리뷰 3~4건 0.299 → 20건 이상 0.472), 콘텐츠 기반은 이력이 적은 사용자에서 상대적으로 낫습니다.
- **회귀 기반 사용자 가중치(02)는 전역 가중치보다 낫지 않았고**, 페르소나 간 차이도 거의 없었습니다(중요도 비율 차이 약 0.003).
- **편차 기반(04·05)에서는 관대함이 뚜렷한 신호입니다** (0.905 → 0.878). aspect별 개인 까다로움도 있지만 효과는 작고(−0.001), 페르소나는 예측에 거의 기여하지 않습니다(−0.0002). 대신 "서비스 깐깐형 (위치엔 관대)" 35%, "수면 깐깐형 (서비스엔 관대)" 65%처럼 해석할 수 있는 유형이 나옵니다.
- **aspect 가중치는 플랫폼을 넘어 일반화됩니다.** HotelRec 전역 가중치로 매긴 Booking.com 호텔 점수는 동일 가중보다 실제 점수와 상관이 높습니다(0.274 → 0.301).
- Booking.com 리뷰에서도 여행 유형별 선호 차이가 보입니다. 가족 여행객은 location을, 출장객은 rooms·sleep_quality·value를 더 언급합니다. 키워드 사전을 수정해도 여행 유형별 최다 언급 aspect는 8개 그룹 중 7개, 최소 언급 aspect는 8개 모두 유지됩니다(10).
- 여행 유형과 페르소나의 연관은 회귀 기반·편차 기반 모두 약합니다(Cramér's V 0.03~0.05).
- **HotelRec에서 배운 호텔 점수는 Booking.com 호텔 평균 점수와 상관 0.74**로, 다른 플랫폼의 신규 사용자에게도 호텔 수준의 예측은 통합니다.

## 7. 한계와 남은 과제

- **하이브리드 결합 방식**: 지금 결합(MF + Content)은 순위에서 효과가 없습니다. CF 쪽을 ItemKNN으로 바꾸거나 인기도를 함께 넣는 방안을 검토해야 합니다.
- **페르소나 활용**: 편차 기반 페르소나(05의 `user_personas_dev_tuned.csv`)는 아직 06의 PersonaPopularity·콘텐츠 기반에 연결되지 않았습니다. 예측 기여가 작아 효과는 크지 않을 것으로 보입니다.
- **평가 범위**: 같은 도시 평가는 대도시 사용자(표본의 25%)만 포함합니다. Kaggle 매칭은 6개 유럽 도시 505개 호텔에 한정됩니다.
- **수작업 규칙**: aspect 키워드 사전(08, 수정안은 10)과 호텔명 매칭 규칙(11)은 직접 정한 것입니다. 매칭 목록은 `outputs/11_matched_hotels.csv`에서 확인할 수 있습니다.
- **편차의 기준**: 04의 호텔 평균에는 사용자 자신의 train 리뷰도 포함되어, 리뷰가 적은 호텔에서는 편차가 약간 작게 잡힙니다.

## 8. 참고 문헌

- Antognini & Faltings (2020) HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset. LREC.
- Sarwar et al. (2001) Item-based collaborative filtering recommendation algorithms. WWW.
- Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS.
- Koren, Bell, Volinsky (2009) Matrix factorization techniques for recommender systems. IEEE Computer.
- Koren (2010) Factor in the neighbors: Scalable and accurate collaborative filtering. ACM TKDD.
