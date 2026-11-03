# Modeling pipeline

Trains and evaluates the four filtering methods of the proposal on the preprocessed data (`data/processing/`). It estimates users' aspect preferences and traveler personas, and checks whether they also work for new users on another platform (Booking.com).

| Proposal | Implementation |
| --- | --- |
| ① Memory-based CF | Item-kNN (`models.ItemKNN`) |
| ② Model-based CF | biased matrix factorization, L2 regularization, SGD (`models.BiasedMF`) |
| ③ Content-based | user aspect importance × hotel aspect means + review TF-IDF similarity (`models.ContentBased`) |
| ④ Hybrid | rating: λ · MF + (1 − λ) · content-based / ranking: λ · Item-kNN + (1 − λ) · content-based (after per-user standardization) |
| Linear regression · K-means | user aspect importance estimation, traveler personas |

- **HotelRec (TripAdvisor)**: model design, training and evaluation (each user's latest review is the test)
- **515K Hotel Reviews (Booking.com, Kaggle)**: new-user generalization test (no user IDs, so CF cannot be evaluated directly)

## 1. Overview

```text
data/processing/hotelrec/
  │
  ├─ 01_split ─────────────────── train/validation/test split
  ├─ 02_user_aspect_weights ───── regression-based aspect weights, personas
  ├─ 03_hotel_features ────────── hotel aspect means, TF-IDF
  ├─ 04_persona_deviation ─────── deviation-based user preferences (leniency, strictness), personas
  ├─ 05_persona_tuning ────────── settings selection for 04 and test evaluation
  ├─ 06_train_evaluate ────────── recommendation model training and evaluation (models.py)
  └─ 07_tune_itemknn (optional) ─ evidence for the Item-kNN settings
data/processing/kaggle/
  ├─ 08_kaggle_generalization ─── checks trip-type preferences via aspect mentions in reviews
  ├─ 09_lexicon_audit ─────────── keyword lexicon audit
  ├─ 10_persona_lexicon_compare ─ persona and keyword lexicon comparison
  └─ 11_kaggle_hotel_matching ─── predicted vs actual scores on shared hotels
```

Run in numeric order and each step gets the inputs it needs from the previous ones. Intermediate artifacts go to `data/pipeline/` (excluded from Git), result tables (csv) and figures (png) to `Pipeline/outputs/`; the number in front of a result file name is the number of the script that made it.

## 2. Setup

**Libraries**: install with `requirements` in the repository root (the pipeline uses numpy, pandas, scikit-learn and matplotlib; scipy is installed together with scikit-learn).

**Data**: place the preprocessing results from the team Drive as below.

```text
data/processing/hotelrec/  hotelrec_train.csv, hotelrec_partial.csv, hotel_map.csv, user_map.csv
data/processing/kaggle/    kaggle_test.csv, hotel_meta_kaggle.csv
```

> `hotel_map.csv` must be the **46,747-row** version. The 162,023-row version shared first does not match the `hotel_id`s of the review data. With it, the same-city evaluation in 06 is skipped automatically and the hotel matching in 11 is wrong.

## 3. Running

Run from the `Pipeline/` folder in numeric order. About 30 minutes in total.

```bash
cd Pipeline
python 01_split.py                    # about 0.5 min
python 02_user_aspect_weights.py      # about 2 min
python 03_hotel_features.py           # about 4 min
python 04_persona_deviation.py        # about 1 min
python 05_persona_tuning.py           # about 0.5 min
python 06_train_evaluate.py           # about 10 min
python 07_tune_itemknn.py             # optional, about 9 min
python 08_kaggle_generalization.py    # about 0.5 min
python 09_lexicon_audit.py            # about 1 min
python 10_persona_lexicon_compare.py  # about 0.5 min
python 11_kaggle_hotel_matching.py    # about 0.1 min
```

`06_train_evaluate.py` and `04_persona_deviation.py` take options to change their settings (`python 06_train_evaluate.py -h`).

## 4. Files

| File | Role | Input (earlier steps) | Main outputs |
| --- | --- | --- | --- |
| `common.py` | paths, data loading, RMSE/MAE, HR@k/NDCG@k, Top-N candidate generation | – | – |
| `models.py` | recommendation models (same `fit` / `predict` interface) | – | – |
| `01_split.py` | per-user chronological split | preprocessing results | `01_split_summary.csv` |
| `02_user_aspect_weights.py` | regression-based aspect weights, personas | 01 | `02_*.csv`, `figures/02_*.png` |
| `03_hotel_features.py` | hotel aspect means, hotel TF-IDF | 01 | `03_hotel_top_terms.csv` |
| `04_persona_deviation.py` | deviation-based leniency and strictness, personas | 01, 02, 03 | `04_*.csv`, `figures/04_*.png` |
| `05_persona_tuning.py` | selection of 04's shrinkage strength, leniency coefficients and K; test evaluation | 01~04 | `05_*.csv`, `figures/05_*.png` |
| `06_train_evaluate.py` | model training, rating prediction and Top-N evaluation, plugging in the deviation-based preferences, rating and ranking hybrids | 01, 02, 03, 05 | `06_*.csv`, `figures/06_*.png` |
| `07_tune_itemknn.py` | Item-kNN settings comparison (optional) | 01 | `07_itemknn_tuning.csv` |
| `08_kaggle_generalization.py` | aspect mention analysis of Kaggle reviews, keyword lexicon (`LEXICON`) | 02 | `08_*.csv`, `figures/08_*.png` |
| `09_lexicon_audit.py` | per-keyword frequency, co-mention and sensitivity audit | 08's lexicon | `09_*.csv` |
| `10_persona_lexicon_compare.py` | 3 persona sets × 2 lexicons comparison, lexicon robustness | 02, 04, 05, 08 | `10_*.csv` |
| `11_kaggle_hotel_matching.py` | predicted vs actual scores on hotels shared by HotelRec and Kaggle | 02, 03, 06 | `11_*.csv`, `figures/11_*.png` |

## 5. Methods by step

### 01 Split
From all reviews (train with complete sub-ratings plus partial), **each user's most recent review goes to test, the one before it to val, and the rest to train**. There is no time leakage of predicting the past from future reviews. Dates are monthly, so reviews in the same month are ordered with seed-42 random numbers.

### 02 Regression-based aspect weights
- Global linear regression: `rating ≈ w0 + Σ w_k · aspect_k` (train reviews with all sub-ratings)
- Per-user Ridge: users have only 3.7 complete reviews on average, so the regularization **pulls the coefficients toward the global weights** instead of toward 0. The strength λ is chosen on val.
- Personas: the weights are turned into importance shares and clustered with K-means (K chosen by silhouette score).

### 03 Hotel features
- Hotel aspect means: the sub-ratings of train reviews averaged per aspect; hotels with few reviews are shrunk toward the overall mean.
- Hotel TF-IDF: the titles and texts of train reviews are pooled per hotel to compute TF-IDF, keeping only the top 100 terms per hotel.

### 04·05 Deviation-based user preferences and personas
02's regression coefficients did not distinguish users because there are too few reviews. So instead of estimating coefficients, this averages **how much higher or lower the user's aspect scores are than that hotel's mean aspect scores**.
- **Leniency** L_u: how much more generously (+) or harshly (−) the user scores than the hotel mean across all aspects
- **Strictness**: the sign-flipped per-aspect relative deviation after removing leniency. The larger it is, the more harshly the user rates that aspect compared with others. Aspects with few reviews are shrunk toward 0 (M).
- Personas: the strictness vectors are clustered with K-means (e.g. "Strict on service").
- Validation: RMSE of predicting the aspect scores of val/test reviews as `hotel mean + leniency (+ persona or personal strictness)`. 05 chooses the shrinkage strength M, per-aspect leniency coefficients β and K on val, and reports the final test result.

### 06 Recommendation models and evaluation

| Model | Method |
| --- | --- |
| Popularity | hotels with the most reviews (baseline) |
| PersonaPopularity (regression / deviation) | hotels rated 4+ by the same persona (02's K=3 / 05's K=2) (the proposal's cluster baseline) |
| Baseline | μ + b_u + b_i |
| ItemKNN | rating prediction: baseline prediction (μ+b_u+b_i) + weighted mean of the neighbor hotels' residuals (KNNBaseline)<br>Top-N: sum of co-visit similarities to the hotels the user has visited |
| MF | μ + b_u + b_i + p_u·q_i, MSE + L2, mini-batch SGD, early stopping on val |
| Content | aspect utility (02's user weights × hotel aspect means) + γ × TF-IDF cosine similarity |
| Content (strictness) | content-based with the deviation-based preferences from 04·05 plugged in. The aspect weights are tilted toward the aspects the user is strict about (w_k ∝ global weight × exp(τ · strictness)) and κ × leniency is added to the intercept. Strictness is compared three ways: personal / persona mean / persona only for users with little history (fewer than 5 train reviews) |
| Hybrid-rating (MF+Content) | λ · MF + (1 − λ) · Content(strictness, persona if few reviews). Both are on the rating scale and are summed as is; λ chosen by val RMSE |
| Hybrid-ranking (ItemKNN+Content) | λ · z(ItemKNN co-visit score) + (1 − λ) · z(Content(strictness, persona if few reviews)). z standardizes among the same user's candidates (scale matching); λ chosen by val NDCG@10 |

τ (strength of the strictness tilt) is chosen by val NDCG@10, κ (leniency coefficient) by least squares on the val rating residuals (`06_strictness_tau.csv`).

Why the hybrid is split into a rating one and a ranking one: with a single MF + content-based hybrid and λ chosen by val NDCG@10, both models are weak at ranking and λ=0 (content only) was chosen. So rating prediction is combined with MF and ranking with Item-kNN, the CF that is strongest at ranking, and each λ is chosen by its own target metric (`06_hybrid_lambda.csv`).

Evaluation metrics:
- **Rating prediction**: RMSE and MAE on the test reviews of all users
- **Top-N**: HR@k and NDCG@k with 1 correct hotel (the hotel of the test review) mixed with candidates the user has not rated.
  - Random candidates: 99 hotels from anywhere in the world (sample of 20,000 test users)
  - Same-city candidates: 49 hotels in the same city as the correct one. This matches "choosing a hotel after choosing the destination"; only users in cities with 50+ hotels are evaluated.

07 compares Item-kNN's baseline prediction type, number of neighbors k and shrinkage strength on val, recording why 06's defaults (k=2000, shrink=100, damp=0.03) were chosen.

### 08~10 Kaggle text validation
Each Kaggle review is treated as one new user. A per-aspect keyword lexicon (`LEXICON` in `08`) finds the aspects mentioned in the positive and negative reviews.
- 08: differences in aspect mentions by trip type, correspondence with the HotelRec personas, correlation of hotel scores from HotelRec weights with the actual scores
- 09: lexicon audit. Frequency per keyword, how often two aspects are caught in the same review, and how much the results change when keywords are removed one at a time
- 10: compares the revised lexicon (reflecting 09's audit) with the original, and the regression-based with the deviation-based personas, on the same metrics, and checks whether the conclusions hold when the lexicon changes

### 11 Kaggle hotel matching
Finds the same hotel in both datasets by normalizing hotel names within the same city. Only exact matches or string similarity of 0.9 or more are accepted. Then it evaluates how well the HotelRec models' **new-user predictions** (user terms 0) match the actual Booking.com scores.

## 6. Main results

With seed 42. The full tables are in `outputs/`.

**Rating prediction (06, test)**

| Model | RMSE | MAE |
| --- | --- | --- |
| GlobalMean | 1.004 | 0.784 |
| Baseline | 0.909 | 0.700 |
| ItemKNN | 0.909 | 0.700 |
| MF | 0.907 | **0.693** |
| Content (aspect + text) | 0.952 | 0.745 |
| **Hybrid-rating (MF 0.8 + Content 0.2)** | **0.906** | 0.694 |

**Top-N (06, test HR@10 / NDCG@10)**

| Model | Random candidates (1+99) | Same-city candidates (1+49) |
| --- | --- | --- |
| **ItemKNN** | **0.349 / 0.211** | 0.392 / 0.191 |
| Hybrid-ranking (ItemKNN 0.8 + Content 0.2) | 0.347 / **0.211** | 0.389 / 0.191 |
| Popularity | 0.328 / 0.176 | **0.479 / 0.255** |
| Baseline | 0.146 / 0.073 | 0.312 / 0.152 |
| Content (aspect + text) | 0.135 / 0.065 | 0.296 / 0.143 |
| MF | 0.117 / 0.053 | 0.285 / 0.131 |
| Random ranking | 0.100 | 0.200 |

**User aspect preferences (05, test sub-rating prediction RMSE)**

| Prediction | RMSE |
| --- | --- |
| hotel mean only | 0.905 |
| + leniency | 0.890 |
| + per-aspect leniency (β) | 0.878 |
| + persona (K=2) | 0.878 |
| **+ personal strictness (M=30)** | **0.877** |

**Plugging the deviation-based preferences into recommendation (06, test, τ=160, κ=0.51)**

| Model | Rating RMSE | Random HR@5 / NDCG@10 | Same-city NDCG@10 |
| --- | --- | --- | --- |
| Content (aspect + text) | 0.952 | 0.070 / 0.065 | 0.143 |
| Content (strictness, personal) | 0.945 | 0.076 / 0.067 | 0.142 |
| **Content (strictness, persona)** | **0.931** | 0.077 / 0.068 | **0.143** |
| Content (strictness, persona if few reviews) | 0.939 | **0.078 / 0.068** | 0.142 |
| PersonaPopularity (regression, K=3) | – | 0.215 / 0.172 | 0.251 |
| PersonaPopularity (deviation, K=2) | – | 0.216 / 0.172 | 0.251 |

**505 hotels shared with Kaggle (11, 203,215 reviews): new-user prediction vs actual score**

| Model | Review RMSE | Correlation with hotel mean score |
| --- | --- | --- |
| GlobalMean | 0.812 | – |
| MF (μ + b_i) | 0.784 | 0.743 |
| Content (aspect) | 0.788 | 0.674 |
| **Hybrid-rating (λ=0.8)** | **0.782** | **0.743** |

**Interpretation**
- MF is best at rating prediction, but the gap to Baseline is small. Users have 7 reviews on average, so there is little information to gain from personalization.
- In Top-N, ItemKNN (co-visits) and popularity are strongest. Models that predict ratings (MF, Content) are weak at guessing "which hotel the user will go to".
- ItemKNN improves for users with more history (HR@10: 0.299 with 3~4 reviews → 0.472 with 20+), while content-based does relatively better for users with little history.
- **The regression-based user weights (02) were no better than the global weights**, and the personas barely differed (importance share difference about 0.003).
- **In the deviation-based approach (04·05), leniency is a clear signal** (0.905 → 0.878). Per-aspect personal strictness exists but its effect is small (−0.001), and the personas contribute almost nothing to prediction (−0.0002). In return they give interpretable types such as "Strict on service (lenient on location)" 35% and "Strict on sleep (lenient on service)" 65%.
- **Plugging the deviation-based preferences into content-based recommendation helps a little.** Rating RMSE went 0.952 → 0.931 (mostly the leniency correction), random-candidate HR@5 0.070 → 0.078 (+11%), NDCG@10 0.065 → 0.068. There is no difference on same-city candidates. The persona-mean strictness has a lower RMSE than personal strictness, meaning the personal values are noisy and grouping by persona stabilizes them. The largest gain was for users with 20+ reviews, HR@10 0.110 → 0.119. Still, the RMSE stays higher than MF and Baseline (0.907~0.909).
- **Once split by purpose, the hybrids actually mix CF and content (λ=0.8 for both).** The rating hybrid, with test RMSE 0.906, edges past MF (0.907) as the best rating predictor, and is also slightly better than MF for Booking.com new-user prediction (review RMSE 0.784 → 0.782). The ranking hybrid beat ItemKNN alone on val (NDCG@10 0.2283 → 0.2297) but is on par with ItemKNN on test (random NDCG@10 0.211, same city 0.191). For ranking, content-based adds almost no information on top of ItemKNN.
- PersonaPopularity, both regression-based and deviation-based, is slightly below popularity (0.328). With only 2~3 personas, the popular hotels per cluster are almost the same as the overall popular hotels.
- **The aspect weights generalize across platforms.** Booking.com hotel scores from the HotelRec global weights correlate more with the actual scores than equal weights do (0.274 → 0.301).
- Booking.com reviews also show preference differences by trip type. Families mention location more; business travelers mention rooms, sleep_quality and value more. With the revised keyword lexicon, the most-mentioned aspect per trip type stays the same in 7 of 8 groups and the least-mentioned in all 8 (10).
- The association between trip type and persona is weak for both the regression-based and deviation-based personas (Cramér's V 0.03~0.05).
- **The hotel scores learned on HotelRec correlate 0.74 with the Booking.com hotel mean scores**, so hotel-level prediction works even for new users on another platform.

## 7. How the methods were improved

What problems showed up with the first approaches, and how they were changed to reach the current values. All settings were chosen on val; test was reported separately.

**① Data mapping (preprocessing)**

| Stage | What | Result |
| --- | --- | --- |
| First | used `hotel_map.csv` (162,023 rows) from the team Drive | the row count already differs from the 46,747 hotels in the review data. The city name appears in that hotel's reviews only **3.9%** of the time (random level) |
| Cause | a mapping from a different run, before the sampling step | `user_map.csv` was also missing, so the validation script could not run |
| Fix | reran the preprocessing code on the original `HotelRec.txt` | train/partial **byte-identical** to the shared files → only the 2 mappings replaced |
| Now | 46,747-row `hotel_map.csv`, 130,036-row `user_map.csv` | city name match **87.9%**, validation **24/24 PASS**. The same-city evaluation (06) and Kaggle matching (11) became possible |

**② Top-N evaluation (06)**

| Stage | What | Result |
| --- | --- | --- |
| First | 1 correct hotel + 99 random hotels from around the world | the popularity baseline reached HR@10 **0.328**, more than twice every personalized model at the time (MF 0.117, content 0.135) |
| Cause | random candidates are mostly unpopular hotels in other regions, easy to tell apart by popularity alone | also differs from the proposal's setting of "choosing a hotel within the same region" |
| Fix | after fixing the mapping in ①, added the **same-city candidates** (1+49) evaluation | 4,975 users in cities with 50+ hotels |
| Now | both are reported | popularity is strongest in the same city too (0.479), then ItemKNN (0.392) |

**③ Memory-based CF: Item-kNN (06, 07)**

| Stage | Setting | val RMSE | val HR@10 | Note |
| --- | --- | --- | --- | --- |
| First | adjusted cosine (user-mean baseline, as in the proposal), k=50 | 1.012 | 0.049 | worse than predicting the mean (0.974) |
| Fix 1 | baseline prediction μ+b_u+b_i (KNNBaseline), stronger shrinkage | **0.877** | 0.157 | reflects the hotel effect when there are no neighbors |
| Fix 2 | separate ranking score: sum of co-visit similarities to the hotels the user has visited | 0.877 | 0.216 | the rating predictions tie for most candidates and cannot rank them |
| Fix 3 | number of neighbors k = 50 → 200 → 500 → **2000** | 0.877 | 0.280 → 0.321 → **0.371** | share of neighbors overlapping with the user's history 7.4% → 22.7% |
| Now (test) | Fix 3 setting | RMSE 1.034 → **0.909** | HR@10 0.035 → **0.349** | beats popularity (0.328) on random candidates for the first time |

**④ User aspect preferences and personas (02, 04, 05, 06)**

| Stage | Method | Result |
| --- | --- | --- |
| First (02) | per-user aspect weights from linear regression (Ridge shrinking toward the global weights), K-means personas | the stronger the personalization, the larger the val error (λ=10: 0.515, global: 0.457). At the chosen λ=1000 the personal weights ≈ global weights, and the 3 personas differ by about **0.003** |
| Cause | users have only 3.7 reviews with complete sub-ratings on average, and the aspects are highly correlated | not enough data to estimate 7 coefficients per user |
| Fix 1 (04) | instead of regression, average the **deviation from the hotel mean**: leniency + per-aspect strictness | test sub-rating RMSE 0.905 (hotel mean) → 0.890 (+leniency). Adding personal strictness made it worse, 0.896 |
| Fix 2 (05) | strictness shrinkage strength M=3 → **30**, per-aspect leniency coefficients β, K comparison | 0.878 (+β leniency) → **0.877** (+personal strictness). 2 interpretable personas: Strict on service 35%, Strict on sleep 65% |
| Fix 3 (06) | plug strictness into content-based recommendation (τ candidates 0~80 → optimum at the boundary, extended to 0~320 → **τ=160**) | content-based RMSE 0.952 → **0.931**, random HR@5 0.070 → **0.078** |
| Now | the deviation-based preferences feed content-based, the hybrids and PersonaPopularity | consistent but small improvements. No difference on same-city candidates |

**⑤ Hybrid (06, 11)**

| Stage | Method | Result |
| --- | --- | --- |
| First | λ · MF + (1 − λ) · content, λ chosen by val NDCG@10 | **λ=0** chosen → the hybrid equals content-based |
| Cause | MF and content are both weak at ranking, so choosing by a ranking metric gives no reason to mix | in the same table, val RMSE with 90% MF (0.8757) was lower than MF alone (0.8760) |
| Fix | split by purpose: **rating** = MF + content (λ by val RMSE), **ranking** = ItemKNN + content (after per-user standardization, λ by val NDCG@10) | both hybrids actually mix at **λ=0.8** |
| Now (test) | rating hybrid / ranking hybrid | rating RMSE **0.906** (MF 0.907, best), Kaggle review RMSE 0.784 → 0.782. Ranking NDCG@10 0.211, on par with ItemKNN alone |

**⑥ Kaggle generalization check (08~11)**

| Stage | Method | Result |
| --- | --- | --- |
| First (08) | count aspect mentions in reviews with a hand-made keyword lexicon and compare with trip types and personas | clear differences by trip type (family → location, business → rooms·sleep·value) but a weak association with personas (Cramér's V 0.034) |
| Audit (09) | frequency and sensitivity per keyword | the two words `location` and `clean` account for about 79% of their aspects' mentions; removing one changes the result by up to 4.7pp |
| Fix (10) | re-compare with the revised lexicon (4 removed, 21 added) and the deviation-based personas | the most-mentioned aspect per trip type kept in 7 of 8 groups, the least-mentioned in all 8. The persona association is still weak (V 0.03~0.05) |
| Extra (11) | text comparison alone cannot show the models' generalization directly, so the **same hotels** in both datasets were matched | on 505 hotels, the HotelRec hotel scores correlate **0.74** with the Booking.com hotel means |

## 8. Limitations and remaining work

- **Effect of the hybrids**: the rating hybrid improves by about 0.001, and the ranking hybrid does not differ from ItemKNN alone on test. Popularity is still strongest on same-city candidates, so combining popularity as well remains an option.
- **Use of personas**: the deviation-based preferences are plugged into 06's content-based model, the hybrids and PersonaPopularity, but the improvements are small. τ=160 is a strong setting that puts about 89% of the weight on the single aspect the user is strictest about, so it should be read as "choosing hotels mainly on the strictest aspect". They are not plugged into 11's Kaggle new-user prediction because there is no user history.
- **Evaluation coverage**: the same-city evaluation covers only users in large cities (25% of the sample). The Kaggle matching is limited to 505 hotels in 6 European cities.
- **Hand-made rules**: the aspect keyword lexicon (08, revised in 10) and the hotel name matching rules (11) were set by hand. The match list can be checked in `outputs/11_matched_hotels.csv`.
- **Deviation baseline**: the hotel means in 04 include the user's own train reviews, so for hotels with few reviews the deviation comes out slightly smaller.

## 9. References

- Antognini & Faltings (2020) HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset. LREC.
- Sarwar et al. (2001) Item-based collaborative filtering recommendation algorithms. WWW.
- Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS.
- Koren, Bell, Volinsky (2009) Matrix factorization techniques for recommender systems. IEEE Computer.
- Koren (2010) Factor in the neighbors: Scalable and accurate collaborative filtering. ACM TKDD.
