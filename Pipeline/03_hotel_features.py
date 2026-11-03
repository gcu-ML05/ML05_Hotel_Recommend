"""
Step 3: hotel features for content-based filtering

(1) Hotel aspect means (n_hotels, 6)
    Average the sub-ratings of train reviews (complete + partial), per aspect, over the available values.
    To keep hotels with few reviews from jumping around, shrink toward the overall mean (Bayesian mean).
        a_hk = (Σ scores + M · overall_mean_k) / (count + M)

(2) Hotel TF-IDF (n_hotels, vocab)
    Apply TF-IDF to the word counts of train-review titles + bodies grouped by hotel (val/test text is not used).
    To avoid loading all text (about 1GB) at once, the vocabulary is built from a 100k-review sample and the
    counts are accumulated chunk by chunk.
    Each hotel keeps only its TOP_TERMS highest TF-IDF terms (re-normalized with L2) to keep similarity cheap.

Output: data/pipeline/hotel_aspects.npy, hotel_tfidf.npz, vocab.txt
        Pipeline/outputs/03_hotel_top_terms.csv
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from common import ASPECTS, HOTELREC_DIR, SEED, WORK_DIR, ensure_dirs, load_split, save_table

M = 5                 # shrinkage strength of aspect means (number of pseudo-reviews)
VOCAB_SAMPLE = 100_000
MAX_FEATURES = 10_000
TOP_TERMS = 100
CHUNK = 100_000


def hotel_aspect_means(tr, n_hotels):
    A = np.zeros((n_hotels, len(ASPECTS)))
    for k, c in enumerate(ASPECTS):
        s = tr[["hotel_id", c]].dropna()
        cnt = np.bincount(s.hotel_id, minlength=n_hotels)
        tot = np.bincount(s.hotel_id, weights=s[c], minlength=n_hotels)
        mu = s[c].mean()
        A[:, k] = (tot + M * mu) / (cnt + M)
    return A


def iter_train_text(train_keys, n_hotels):
    """Read the source CSVs in chunks and yield (hotel_id, title+body) only for reviews in the train split"""
    for f in ("hotelrec_train.csv", "hotelrec_partial.csv"):
        for ch in pd.read_csv(HOTELREC_DIR / f, usecols=["user_id", "hotel_id", "title", "text"],
                              chunksize=CHUNK, encoding="utf-8-sig"):
            key = ch.user_id.to_numpy(np.int64) * n_hotels + ch.hotel_id.to_numpy(np.int64)
            ch = ch[np.isin(key, train_keys)]
            yield ch.hotel_id.to_numpy(), (ch.title.fillna("") + " " + ch.text.fillna("")).tolist()


def keep_top_terms(T, k):
    """Keep only the k largest terms in each row (hotel) and re-normalize with L2"""
    T = T.tocsr()
    rows, cols, vals = [], [], []
    for i in range(T.shape[0]):
        s, e = T.indptr[i], T.indptr[i + 1]
        d, idx = T.data[s:e], T.indices[s:e]
        if len(d) > k:
            top = np.argpartition(-d, k)[:k]     # find the top-k positions without a full sort
            d, idx = d[top], idx[top]
        rows.append(np.full(len(d), i))
        cols.append(idx)
        vals.append(d)
    out = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=T.shape, dtype=np.float32)
    norm = np.sqrt(out.multiply(out).sum(1)).A1
    return sp.diags(1 / np.where(norm > 0, norm, 1)) @ out


def main():
    ensure_dirs()
    df, n_users, n_hotels = load_split()
    tr = df[df.split == "train"]

    # (1) aspect means
    A = hotel_aspect_means(tr, n_hotels)
    np.save(WORK_DIR / "hotel_aspects.npy", A)
    print("saved hotel aspect means:", A.shape, "mean", dict(zip(ASPECTS, A.mean(0).round(3))))

    # (2) TF-IDF
    train_keys = np.sort(tr.user_id.to_numpy(np.int64) * n_hotels + tr.hotel_id.to_numpy(np.int64))
    rng = np.random.default_rng(SEED)
    p = VOCAB_SAMPLE / len(tr)
    sample = []
    for _, texts in iter_train_text(train_keys, n_hotels):
        pick = rng.random(len(texts)) < p
        sample += [t for t, m in zip(texts, pick) if m]
    # CountVectorizer: builds a document-term count matrix
    #   stop_words="english": drop stop words such as "the", "and" / max_features: keep only the most frequent terms
    #   min_df=5: drop terms appearing in fewer than 5 documents / token_pattern: alphabetic words of 2+ letters (no digits)
    cv = CountVectorizer(stop_words="english", max_features=MAX_FEATURES, min_df=5,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", dtype=np.float32)
    cv.fit(sample)
    vocab = cv.get_feature_names_out()
    print(f"vocabulary: sample of {len(sample):,} reviews -> {len(vocab):,} terms")
    del sample

    C = sp.csr_matrix((n_hotels, len(vocab)), dtype=np.float32)
    n_docs = 0
    for hid, texts in iter_train_text(train_keys, n_hotels):
        X = cv.transform(texts)                                   # (n_reviews, n_terms)
        H = sp.csr_matrix((np.ones(len(hid), np.float32), (np.arange(len(hid)), hid)),
                          shape=(len(hid), n_hotels))             # review -> hotel indicator matrix
        C = C + H.T @ X                                           # accumulate term counts per hotel
        n_docs += len(hid)
        print(f"   ...{n_docs:,} reviews processed")
    # TfidfTransformer: multiplies the count matrix by IDF weights
    #   sublinear_tf=True: 1+log(tf) instead of tf (dampens hotels with many long reviews) / norm="l2": row normalization
    T = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(C)
    T = keep_top_terms(T, TOP_TERMS)
    sp.save_npz(WORK_DIR / "hotel_tfidf.npz", T)
    (WORK_DIR / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    print(f"saved hotel TF-IDF: {T.shape}, nnz {T.nnz:,}")

    # top terms of the 5 most-reviewed hotels (for the presentation/report)
    hm = pd.read_csv(HOTELREC_DIR / "hotel_map.csv", encoding="utf-8-sig").set_index("hotel_id")
    top_h = tr.hotel_id.value_counts().index[:5]
    rows = []
    for h in top_h:
        r = T.getrow(h)
        order = np.argsort(-r.data)[:10]
        rows.append({"hotel_id": h, "hotel_name": hm.at[h, "hotel_name"], "location": hm.at[h, "location"],
                     "top_terms": ", ".join(vocab[r.indices[order]])})
    top = pd.DataFrame(rows)
    print(top.to_string(index=False))
    save_table(top, "03_hotel_top_terms")


if __name__ == "__main__":
    main()
