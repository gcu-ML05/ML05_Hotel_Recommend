# Hotel Recommender System Term Project — Data Preprocessing

## 1. About the project

This repository manages the preprocessing of the data used in the Machine Learning Team 5 project **A Rating-Based Hybrid "Hotel Recommender System"**.

Two hotel review datasets are analyzed, cleaned and converted into a form that can be used to train and evaluate the recommendation models.

### Project goals

- Understand the structure and properties of the datasets
- Handle missing values, duplicates and invalid data
- Convert the data into a form suited to recommender systems (user ID, hotel ID, rating, sub-ratings, review text)
- Validate the quality of the preprocessing results
- Set up an environment where every team member can reproduce the same process

## 2. Datasets

The two datasets are not merged; **each is preprocessed separately**.

|             | Dataset A: HotelRec                                | Dataset B: 515K Hotel Reviews in Europe                                      |
| ----------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| Purpose     | model design and training                          | new-user generalization test                                                 |
| Source      | TripAdvisor reviews. Antognini & Faltings (2020)   | Booking.com reviews. Kaggle                                                  |
| Link        | https://github.com/Diego999/HotelRec               | https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe |
| Size (raw)  | 50,264,531 reviews / 21,891,401 users / 365,055 hotels | 515,738 reviews / 1,494 hotels (6 European cities)                       |
| Period      | 2001.02 ~ 2019.05                                  | 2015 ~ 2017                                                                  |
| User ID     | yes (`author`)                                     | no                                                                           |
| Ratings     | overall rating (1~5) + 6 sub-ratings (some reviews only) | `Reviewer_Score` (10-point scale)                                      |
| Review text | title + body                                       | positive/negative separated                                                  |
| Other       | -                                                  | trip type tags, nationality, hotel coordinates                               |
| Terms of use | **academic research only**                        | Kaggle dataset                                                               |

> The raw data is not included in the repository because of its size and terms of use. Download it from the links above and put it in `data/raw/`.
> HotelRec is for academic research, so do not share the preprocessing results through public links.

## 3. Project structure

```text
ML05_Hotel_Recommend/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # not pushed to Git
│   ├── raw/                       # raw data (HotelRec.txt, Hotel_Reviews.csv)
│   └── processing/                # preprocessing results
│       ├── hotelrec/
│       └── kaggle/
├── Preprocessing/
│   ├── 01_preprocess_hotelrec.py  # HotelRec preprocessing (training data)
│   ├── 02_preprocess_kaggle.py    # Kaggle preprocessing (test data)
│   ├── 03_validate_preprocessing.py   # automatic validation of the results
│   └── 04_eda.ipynb               # exploratory data analysis
└── reports/                       # before/after comparisons, validation results, EDA figures (pushed to Git)
```

### Main directories

- `data/raw/`: where the raw data is stored
- `data/processing/`: where the preprocessed data is stored
- `Preprocessing/`: preprocessing and validation code and the EDA notebook
- `reports/`: before/after comparison tables, validation results (`validation_report.csv`), EDA figures (`eda_figures/`)

## 4. Development environment

### Prerequisites

- Python 3.10 or later
- Git
- A code editor such as Visual Studio Code (to run the notebook)

### Step 1. Clone the repository

```bash
git clone [GITHUB_REPOSITORY_URL]
cd ML05_Hotel_Recommend
```

Replace `[GITHUB_REPOSITORY_URL]` with the actual GitHub repository address.

### Step 2. Create a virtual environment

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

### Step 3. Install the libraries

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Data preprocessing

### 5-1. How to run

Put the raw data in `data/raw/`, then run the scripts in order from the `Preprocessing/` folder.

```bash
cd Preprocessing

# 1) HotelRec (training data)
python 01_preprocess_hotelrec.py --src ../data/raw/HotelRec.txt --out-dir ../data/processing/hotelrec

# 2) Kaggle (test data)
python 02_preprocess_kaggle.py --src ../data/raw/Hotel_Reviews.csv --out-dir ../data/processing/kaggle

# 3) Validate the results (everything must PASS)
python 03_validate_preprocessing.py --hotelrec-dir ../data/processing/hotelrec --kaggle-dir ../data/processing/kaggle
```

For the EDA, open `04_eda.ipynb`, check the paths in the first code cell, then `Run All`.

**Notes on the HotelRec preprocessing**

- It reads the raw file (about 50 million lines) twice, so it takes a long time and creates an intermediate file (`interim_candidates.csv`, tens of GB). **Check the free disk space first.** If it is not enough, point `--out-dir` to another drive.
- To rerun only step 3 after the scans are done, reuse the intermediate file.

```bash
python 01_preprocess_hotelrec.py --resume --interim <path to interim_candidates.csv> --out-dir <output path> \
    --raw-total 50264531 --raw-placeholder 1216923 --raw-users 21891400 --raw-hotels 365055
```

| Option                       | Default | Description                                                              |
| ---------------------------- | ------- | ------------------------------------------------------------------------ |
| `--min-user`                 | 5       | minimum reviews per user                                                 |
| `--min-hotel`                | 20      | minimum reviews per hotel                                                |
| `--max-users`                | 200000  | maximum number of users kept at random after filtering (0 = no sampling) |
| `--min-hotel-after-sampling` | 10      | minimum reviews per hotel for the refiltering after sampling             |
| `--seed`                     | 42      | random seed for the sampling                                             |

### 5-2. Preprocessing criteria

**Team decisions**

| Item                                       | Decision                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| Region                                     | no restriction (worldwide)                                                     |
| Minimum reviews per user/hotel             | users 5+, hotels 20+ (hotels 10+ after user sampling)                          |
| Reviews missing some sub-ratings           | not removed; saved separately (`hotelrec_partial.csv`)                         |
| Duplicate ratings of a hotel by one user   | only the most recent review is kept                                            |
| Kaggle rating scale                        | `Reviewer_Score` divided by 2 to get a 5-point scale                           |

**HotelRec processing order**

1. Remove anonymous authors: `/undefined`, `null`, `A TripAdvisor Member` etc. are not real users and would be grouped into one "giant user", so they are excluded (1,216,923 reviews)
2. Extract candidates: read the raw file one line at a time and save only rows with users of 5+ and hotels of 20+ reviews to an intermediate file
3. Format validation: rating in 1~5, date conversion, missing author/hotel checks. Sub-ratings outside 1~5 are set to missing
4. Duplicate removal: keep only the latest review per (user, hotel)
5. Iterative filtering: repeat the user and hotel conditions until nothing more is removed
6. User sampling: draw at most 200,000 users at random (seed=42), then refilter with the 10-review hotel threshold
7. ID assignment: integer `user_id` and `hotel_id` (the same in train and partial)
8. Split save: `hotelrec_train.csv` if all 6 sub-ratings are present, otherwise `hotelrec_partial.csv`

**Kaggle processing order**

1. Remove exact duplicate rows (526)
2. Date conversion, nationality whitespace cleanup, city extraction from the address
3. `rating = Reviewer_Score / 2` (the original score is kept as `reviewer_score_10`)
4. Replace the `No Negative` / `No Positive` placeholder phrases with empty values and build `review_text` by joining positive + negative
5. Extract trip purpose, traveler type, number of nights, room type, mobile flag and pet flag from `Tags`
6. Format validation, then save

### 5-3. Preprocessing results (before/after)

**HotelRec**

| Step                                    | Rows       | Users      | Hotels  | Reviews per user |
| --------------------------------------- | ---------- | ---------- | ------- | ---------------- |
| Raw                                     | 50,264,531 | 21,891,401 | 365,055 | -                |
| Anonymous authors removed               | 49,047,608 | 21,891,400 | 365,055 | 2.24             |
| First-pass candidates (users≥5, hotels≥20) | 20,694,926 | 2,030,322 | 251,813 | 10.19          |
| Duplicates removed (latest 1)           | 19,819,559 | 2,030,322  | 251,813 | 9.76             |
| Iterative filtering                     | 17,690,307 | 1,747,203  | 162,023 | 10.12            |
| User sampling + refiltering (final)     | 1,228,619  | 130,036    | 46,747  | 9.45             |
| └ train (all 6 sub-ratings)             | 328,995    | 89,687     | 42,651  | 3.67             |
| └ partial (some/no sub-ratings)         | 899,624    | 127,516    | 46,747  | 7.05             |

Reviews with all 6 sub-ratings are about 26.8% of the final data. The missing rate per sub-rating is service 27.3%, cleanliness 49.5%, value 49.5%, rooms 51.5%, location 51.3%, sleep quality 56.9%.

**Kaggle**

| Item                        | Before  | After                    |
| --------------------------- | ------- | ------------------------ |
| Rows                        | 515,738 | 515,212                  |
| Hotels                      | 1,494   | 1,494                    |
| Exact duplicate rows        | 526     | 0                        |
| `No Positive` placeholders  | 35,946  | 0 (converted to empty)   |
| `No Negative` placeholders  | 127,890 | 0 (converted to empty)   |
| `Tags` parse failures       | -       | 0                        |

Missing values: trip purpose 15,004 (2.9%), room type and nights 192 each, reviews with completely empty text 236, coordinates (`lat`/`lng`) 3,268 (17 hotels, kept as is in `hotel_meta_kaggle.csv`).

Traveler type distribution: Couple 252,005 / Solo traveler 108,431 / Group 65,338 / Family with young children 60,975 / Family with older children 26,320 / Travelers with friends 2,143

### 5-4. Output files

**`data/processing/hotelrec/`**

| File                          | Contents                                                                    |
| ----------------------------- | --------------------------------------------------------------------------- |
| `hotelrec_train.csv`          | reviews with all 6 sub-ratings (**not a training/validation split**)        |
| `hotelrec_partial.csv`        | reviews missing some/all sub-ratings                                        |
| `hotel_map.csv`               | `hotel_id`, `hotel_url`, `tripadvisor_id`, `hotel_name`, `location`         |
| `user_map.csv`                | `user_id`, `author`                                                         |
| `reports/hotelrec_report.csv` | before/after comparison per step                                            |

train/partial columns: `user_id, hotel_id, rating, date, service, cleanliness, value, rooms, location, sleep_quality, title, text`

**`data/processing/kaggle/`**

| File                                      | Contents                                                              |
| ----------------------------------------- | --------------------------------------------------------------------- |
| `kaggle_test.csv`                         | cleaned reviews                                                       |
| `hotel_meta_kaggle.csv`                   | `hotel_id`, name, address, city, coordinates, average score, review count |
| `reports/kaggle_report.csv`               | before/after comparison                                               |
| `reports/kaggle_traveler_type_counts.csv` | traveler type distribution                                            |

`kaggle_test.csv` columns: `review_id, hotel_id, hotel_name, city, review_date, reviewer_score_10, rating, reviewer_nationality, trip_type, traveler_type, nights, room_type, is_mobile, with_pet, positive_review, negative_review, review_text`

### 5-5. Known limitations and caveats

- **Users have few reviews with complete sub-ratings.** In train the mean per user is 3.67 (median 3), and 11,160 users (12.4%) have 7 or more. The per-user aspect importance regression needs its own criteria (a higher minimum count, regularization, using partial as well, etc.).
- **No training/validation split was made inside HotelRec.** The evaluation scheme is decided in the modeling step.
- **HotelRec is a random sample of users.** It is 200,000 users drawn from about 1.75 million users who passed the filtering, so reproducing it needs the same seed and the same raw data.
- **Kaggle has no user IDs.** Collaborative filtering cannot be tested; the data is used to validate the content-based model and the traveler types.
- **The rating scale conversion is a simple proportion.** No distribution correction other than 10-point → 5-point (÷2) was applied.
- **Duplicates among short reviews** ("Just perfect" etc.) may have been written by different people, so they were not removed.

## 6. Data management principles

- The raw data is not modified and is kept separately in `data/raw/`.
- The preprocessing results are kept apart from the raw data in `data/processing/`.
- Large data or data with restricted redistribution is not pushed to GitHub. Add the following to `.gitignore`.

```text
data/
*.csv
*.txt
.venv/
```

- The preprocessed data is shared through Google Drive: **(add the Drive link)**
- Copy the before/after comparison tables and validation results (small CSVs) into `reports/` and push them with `git add -f`.
- The rules and assumptions applied in preprocessing are recorded in sections 5-2 and 5-5 of this document.

## 7. Reproducibility

- `requirements.txt` manages the required libraries.
- Random sampling is fixed with `--seed 42`.
- The data size and main statistics before and after preprocessing are recorded automatically in each `reports/*_report.csv`.
- `03_validate_preprocessing.py` automatically checks duplicates, value ranges, minimum review counts and ID consistency; every item must PASS.

## 8. Collaboration rules

1. Pull the latest changes before starting work (`git pull`).
2. Use a separate branch when changing a major feature.
3. Write the preprocessing code split by function.
4. Run and validate with `03_validate_preprocessing.py` before committing.
5. Write commit messages that clearly describe the change. (e.g. `feat: add HotelRec preprocessing code`)
6. When data columns or preprocessing rules change, update this document as well.

## 9. Progress

- [x] Select the 2 datasets and record their sources
- [ ] Initial GitHub repository and development environment setup (`requirements.txt`, `.gitignore`)
- [x] Exploratory data analysis (`04_eda.ipynb`)
- [x] Preprocessing code for each dataset
- [x] Validate the preprocessing results (`03_validate_preprocessing.py`, 24 checks PASS)
- [x] Record the preprocessing results and statistics
- [ ] Prepare the data needed for model development and evaluation (HotelRec training/validation split, train user criteria)

## 10. References

- HotelRec: https://github.com/Diego999/HotelRec — Antognini & Faltings, _HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset_ (2020)
- 515K Hotel Reviews Data in Europe (Kaggle): https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
- Libraries used: pandas, numpy, matplotlib
