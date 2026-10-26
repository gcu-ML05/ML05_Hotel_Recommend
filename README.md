# 호텔 추천 시스템 텀 프로젝트 — 데이터 전처리

## 1. 프로젝트 소개

머신러닝 Team 5 프로젝트 **A Rating-Based Hybrid "Hotel Recommender System"** 에서 사용할 데이터의 전처리 과정을 관리하는 저장소입니다.

두 개의 호텔 리뷰 데이터셋을 분석·정제하여 추천 모델의 학습과 평가에 사용할 수 있는 형태로 변환합니다.

### 프로젝트 목표

- 데이터셋의 구조와 특성 파악
- 결측치, 중복 데이터 및 잘못된 데이터 처리
- 추천 시스템에 적합한 데이터 형태로 변환 (사용자 ID, 호텔 ID, 평점, 세부 평점, 리뷰 텍스트)
- 전처리 결과의 품질 검증
- 팀원 누구나 동일한 과정을 재현할 수 있는 환경 구축

## 2. 데이터셋

두 데이터셋은 서로 합치지 않고 **각각 따로 전처리**합니다.

|             | 데이터셋 A: HotelRec                               | 데이터셋 B: 515K Hotel Reviews in Europe                                     |
| ----------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| 사용 목적   | 모델 설계·학습                                     | 신규 사용자 일반화 테스트                                                    |
| 출처        | TripAdvisor 리뷰. Antognini & Faltings (2020)      | Booking.com 리뷰. Kaggle                                                     |
| 링크        | https://github.com/Diego999/HotelRec               | https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe |
| 규모(원본)  | 리뷰 50,264,531 / 사용자 21,891,401 / 호텔 365,055 | 리뷰 515,738 / 호텔 1,494 (유럽 6개 도시)                                    |
| 기간        | 2001.02 ~ 2019.05                                  | 2015 ~ 2017                                                                  |
| 사용자 ID   | 있음 (`author`)                                    | 없음                                                                         |
| 평점        | 전체 평점(1~5) + 세부 평점 6개(일부 리뷰만)        | `Reviewer_Score` (10점 척도)                                                 |
| 리뷰 텍스트 | 제목 + 본문                                        | 긍정/부정 분리                                                               |
| 기타        | -                                                  | 여행 유형 태그, 국적, 호텔 좌표                                              |
| 이용 조건   | **학술 연구 목적 한정**                            | Kaggle 데이터셋                                                              |

> 원본 데이터는 용량과 이용 조건 때문에 저장소에 포함하지 않습니다. 위 링크에서 직접 받아 `data/raw/`에 넣어주세요.
> HotelRec은 학술 연구용이므로 전처리 결과를 외부에 공개 링크로 공유하지 마세요.

## 3. 프로젝트 구조

```text
ML05_Hotel_Recommend/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # Git에 올리지 않음
│   ├── raw/                       # 원본 (HotelRec.txt, Hotel_Reviews.csv)
│   └── processing/                # 전처리 결과
│       ├── hotelrec/
│       └── kaggle/
├── Preprocessing/
│   ├── 01_preprocess_hotelrec.py  # HotelRec 전처리 (학습용)
│   ├── 02_preprocess_kaggle.py    # Kaggle 전처리 (테스트용)
│   ├── 03_validate_preprocessing.py   # 결과 자동 검증
│   └── 04_eda.ipynb               # 탐색적 데이터 분석
└── reports/                       # 전후 비교, 검증 결과, EDA 그래프 (Git에 올림)
```

### 주요 디렉터리 설명

- `data/raw/`: 원본 데이터 저장 위치
- `data/processing/`: 전처리가 완료된 데이터 저장 위치
- `Preprocessing/`: 전처리·검증 코드와 EDA 노트북
- `reports/`: 전처리 전후 비교표, 검증 결과(`validation_report.csv`), EDA 그래프(`eda_figures/`)

## 4. 개발 환경 설정

### 사전 준비

- Python 3.10 이상
- Git
- Visual Studio Code 등 코드 편집기 (노트북 실행용)

### 1단계. 저장소 복제

```bash
git clone [GITHUB_REPOSITORY_URL]
cd ML05_Hotel_Recommend
```

`[GITHUB_REPOSITORY_URL]`은 실제 GitHub 저장소 주소로 변경합니다.

### 2단계. 가상환경 생성

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3단계. 라이브러리 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. 데이터 전처리 과정

### 5-1. 실행 방법

원본을 `data/raw/`에 넣은 뒤 `Preprocessing/` 폴더에서 순서대로 실행합니다.

```bash
cd Preprocessing

# 1) HotelRec (학습용)
python 01_preprocess_hotelrec.py --src ../data/raw/HotelRec.txt --out-dir ../data/processing/hotelrec

# 2) Kaggle (테스트용)
python 02_preprocess_kaggle.py --src ../data/raw/Hotel_Reviews.csv --out-dir ../data/processing/kaggle

# 3) 결과 검증 (모두 PASS여야 함)
python 03_validate_preprocessing.py --hotelrec-dir ../data/processing/hotelrec --kaggle-dir ../data/processing/kaggle
```

EDA는 `04_eda.ipynb`를 열고 첫 셀의 경로를 확인한 뒤 `Run All` 합니다.

**HotelRec 전처리 주의사항**

- 원본(약 5천만 줄)을 두 번 읽으므로 오래 걸리고, 중간 파일(`interim_candidates.csv`, 수십 GB)이 생깁니다. **디스크 여유 공간을 먼저 확인하세요.** 부족하면 `--out-dir`로 다른 드라이브를 지정합니다.
- 스캔을 마친 뒤 3단계만 다시 돌릴 때는 중간 파일을 재사용합니다.

```bash
python 01_preprocess_hotelrec.py --resume --interim <interim_candidates.csv 경로> --out-dir <저장 경로> \
    --raw-total 50264531 --raw-placeholder 1216923 --raw-users 21891400 --raw-hotels 365055
```

| 옵션                         | 기본값 | 설명                                                        |
| ---------------------------- | ------ | ----------------------------------------------------------- |
| `--min-user`                 | 5      | 사용자 최소 리뷰 수                                         |
| `--min-hotel`                | 20     | 호텔 최소 리뷰 수                                           |
| `--max-users`                | 200000 | 필터링 후 무작위로 남길 최대 사용자 수 (0이면 샘플링 안 함) |
| `--min-hotel-after-sampling` | 10     | 샘플링 후 재필터링에 쓰는 호텔 최소 리뷰 수                 |
| `--seed`                     | 42     | 샘플링 난수 시드                                            |

### 5-2. 전처리 기준

**팀 결정 사항**

| 항목                              | 결정                                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| 지역 범위                         | 제한 없음(전 세계)                                                  |
| 사용자/호텔 최소 리뷰 수          | 사용자 5건 이상, 호텔 20건 이상 (사용자 샘플링 후 호텔은 10건 이상) |
| 세부 평점이 일부 없는 리뷰        | 제거하지 않고 별도 파일(`hotelrec_partial.csv`)로 분리 저장         |
| 같은 사용자의 같은 호텔 중복 평가 | 가장 최근 리뷰만 유지                                               |
| Kaggle 평점 척도                  | `Reviewer_Score`를 2로 나눠 5점 척도로 변환                         |

**HotelRec 처리 순서**

1. 익명 작성자 제거: `/undefined`, `null`, `A TripAdvisor Member` 등은 실제 사용자가 아니라 하나의 "초대형 사용자"로 묶이므로 제외 (1,216,923건)
2. 후보 추출: 원본을 한 줄씩 읽어 사용자 5건 이상, 호텔 20건 이상인 행만 중간 파일에 저장
3. 형식 검증: 평점 1~5 범위, 날짜 변환, 작성자/호텔 결측 확인. 세부 평점이 1~5 밖이면 결측 처리
4. 중복 제거: (사용자, 호텔)별 최신 리뷰 1건만 유지
5. 반복 필터링: 사용자·호텔 조건을 더 이상 줄지 않을 때까지 반복
6. 사용자 샘플링: 사용자를 최대 20만 명 무작위 추출(seed=42)한 뒤 호텔 10건 기준으로 재필터링
7. ID 부여: `user_id`, `hotel_id`를 정수로 부여 (train/partial에서 동일하게 사용)
8. 분리 저장: 6개 세부 평점이 모두 있으면 `hotelrec_train.csv`, 아니면 `hotelrec_partial.csv`

**Kaggle 처리 순서**

1. 완전 중복 행 제거 (526건)
2. 날짜 변환, 국적 공백 제거, 주소에서 도시 추출
3. `rating = Reviewer_Score / 2` (원본 점수는 `reviewer_score_10`으로 보존)
4. `No Negative` / `No Positive` 자리표시 문구를 빈 값으로 바꾸고 긍정+부정을 합친 `review_text` 생성
5. `Tags`에서 여행 목적, 여행자 유형, 숙박일수, 객실 유형, 모바일 여부, 반려동물 동반 여부 추출
6. 형식 검증 후 저장

### 5-3. 전처리 결과 (전후 비교)

**HotelRec**

| 단계                            | 행         | 사용자     | 호텔    | 사용자당 리뷰 |
| ------------------------------- | ---------- | ---------- | ------- | ------------- |
| 원본                            | 50,264,531 | 21,891,401 | 365,055 | -             |
| 익명 작성자 제거                | 49,047,608 | 21,891,400 | 365,055 | 2.24          |
| 1차 후보 (사용자≥5, 호텔≥20)    | 20,694,926 | 2,030,322  | 251,813 | 10.19         |
| 중복 제거 (최신 1건)            | 19,819,559 | 2,030,322  | 251,813 | 9.76          |
| 반복 필터링                     | 17,690,307 | 1,747,203  | 162,023 | 10.12         |
| 사용자 샘플링 + 재필터링 (최종) | 1,228,619  | 130,036    | 46,747  | 9.45          |
| └ train (세부 평점 6개 완전)    | 328,995    | 89,687     | 42,651  | 3.67          |
| └ partial (세부 평점 일부/없음) | 899,624    | 127,516    | 46,747  | 7.05          |

세부 평점 6개를 모두 가진 리뷰는 최종 데이터의 약 26.8%입니다. 세부 평점별 결측률은 service 27.3%, cleanliness 49.5%, value 49.5%, rooms 51.5%, location 51.3%, sleep quality 56.9%입니다.

**Kaggle**

| 항목                   | 전처리 전 | 전처리 후          |
| ---------------------- | --------- | ------------------ |
| 행                     | 515,738   | 515,212            |
| 호텔                   | 1,494     | 1,494              |
| 완전 중복 행           | 526       | 0                  |
| `No Positive` 자리표시 | 35,946    | 0 (빈 값으로 변환) |
| `No Negative` 자리표시 | 127,890   | 0 (빈 값으로 변환) |
| `Tags` 파싱 실패       | -         | 0                  |

결측: 여행 목적 15,004건(2.9%), 객실 유형·숙박일수 각 192건, 본문이 완전히 빈 리뷰 236건, 좌표(`lat`/`lng`) 3,268건(호텔 17곳, `hotel_meta_kaggle.csv`에 그대로 보존).

여행자 유형 분포: Couple 252,005 / Solo traveler 108,431 / Group 65,338 / Family with young children 60,975 / Family with older children 26,320 / Travelers with friends 2,143

### 5-4. 출력 파일

**`data/processing/hotelrec/`**

| 파일                          | 내용                                                                |
| ----------------------------- | ------------------------------------------------------------------- |
| `hotelrec_train.csv`          | 세부 평점 6개가 모두 있는 리뷰 (**학습/검증으로 나눈 분할이 아님**) |
| `hotelrec_partial.csv`        | 세부 평점이 일부/전부 없는 리뷰                                     |
| `hotel_map.csv`               | `hotel_id`, `hotel_url`, `tripadvisor_id`, `hotel_name`, `location` |
| `user_map.csv`                | `user_id`, `author`                                                 |
| `reports/hotelrec_report.csv` | 단계별 전후 비교                                                    |

train/partial 컬럼: `user_id, hotel_id, rating, date, service, cleanliness, value, rooms, location, sleep_quality, title, text`

**`data/processing/kaggle/`**

| 파일                                      | 내용                                                   |
| ----------------------------------------- | ------------------------------------------------------ |
| `kaggle_test.csv`                         | 정제된 리뷰                                            |
| `hotel_meta_kaggle.csv`                   | `hotel_id`, 이름, 주소, 도시, 좌표, 평균 점수, 리뷰 수 |
| `reports/kaggle_report.csv`               | 전후 비교                                              |
| `reports/kaggle_traveler_type_counts.csv` | 여행자 유형 분포                                       |

`kaggle_test.csv` 컬럼: `review_id, hotel_id, hotel_name, city, review_date, reviewer_score_10, rating, reviewer_nationality, trip_type, traveler_type, nights, room_type, is_mobile, with_pet, positive_review, negative_review, review_text`

### 5-5. 알려진 한계와 주의 사항

- **세부 평점 완전 리뷰의 사용자당 건수가 적습니다.** train의 사용자당 평균은 3.67건(중앙값 3건)이고, 7건 이상은 11,160명(12.4%)입니다. 사용자별 aspect 중요도 회귀에는 별도 기준(최소 건수 상향, 정규화, partial 병용 등)이 필요합니다.
- **HotelRec 내부의 학습/검증 분할은 하지 않았습니다.** 평가 방식은 모델링 단계에서 정합니다.
- **HotelRec은 무작위 사용자 샘플입니다.** 필터링을 통과한 사용자 약 175만 명 중 20만 명을 추출한 결과이므로, 재현하려면 같은 시드와 같은 원본이 필요합니다.
- **Kaggle에는 사용자 ID가 없습니다.** 협업 필터링은 테스트할 수 없고, 콘텐츠 기반과 여행자 유형 검증에 사용합니다.
- **평점 척도 변환은 단순 비례입니다.** 10점→5점(÷2) 외의 분포 보정은 하지 않았습니다.
- **짧은 리뷰의 중복**("Just perfect" 등)은 서로 다른 사람이 쓴 것일 수 있어 제거하지 않았습니다.

## 6. 데이터 관리 원칙

- 원본 데이터는 수정하지 않고 `data/raw/`에 별도로 보관합니다.
- 전처리 결과는 원본과 분리하여 `data/processing/`에서 관리합니다.
- 용량이 크거나 재배포가 제한된 데이터는 GitHub에 올리지 않습니다. `.gitignore`에 아래를 추가합니다.

```text
data/
*.csv
*.txt
.venv/
```

- 전처리 결과 데이터는 Google Drive로 공유합니다: **(Drive 링크 추가)**
- 전처리 전후 비교표와 검증 결과(용량이 작은 CSV)는 `reports/`에 복사하고 `git add -f`로 올립니다.
- 전처리 과정에서 적용한 규칙과 가정은 이 문서의 5-2, 5-5에 기록합니다.

## 7. 재현성 관리

- `requirements.txt`로 필요한 라이브러리를 관리합니다.
- 무작위 샘플링은 `--seed 42`로 고정합니다.
- 전처리 전후의 데이터 크기와 주요 통계량은 각 `reports/*_report.csv`에 자동 기록됩니다.
- `03_validate_preprocessing.py`가 중복, 값 범위, 최소 리뷰 수, ID 일관성을 자동으로 확인하며, 모든 항목이 PASS여야 합니다.

## 8. 협업 규칙

1. 작업을 시작하기 전에 최신 변경 사항을 가져옵니다 (`git pull`).
2. 주요 기능을 수정할 때는 별도의 브랜치를 사용합니다.
3. 전처리 코드는 기능별로 분리하여 작성합니다.
4. 커밋하기 전에 실행하고 `03_validate_preprocessing.py`로 검증합니다.
5. 커밋 메시지는 변경 내용을 명확하게 작성합니다. (예: `feat: HotelRec 전처리 코드 추가`)
6. 데이터 컬럼이나 전처리 규칙이 바뀌면 이 문서도 함께 수정합니다.

## 9. 진행 상황

- [x] 데이터셋 2개 선정 및 출처 기록
- [ ] GitHub 저장소 및 개발 환경 초기 설정 (`requirements.txt`, `.gitignore` 반영)
- [x] 탐색적 데이터 분석 수행 (`04_eda.ipynb`)
- [x] 데이터셋별 전처리 코드 구현
- [x] 전처리 결과 검증 (`03_validate_preprocessing.py`, 24개 항목 PASS)
- [x] 전처리 결과 및 통계 기록
- [ ] 모델 개발 및 평가에 필요한 데이터 준비 (HotelRec 학습/검증 분할, train 사용자 기준 결정)

## 10. 참고 자료

- HotelRec: https://github.com/Diego999/HotelRec — Antognini & Faltings, _HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset_ (2020)
- 515K Hotel Reviews Data in Europe (Kaggle): https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
- 사용 라이브러리: pandas, numpy, matplotlib
