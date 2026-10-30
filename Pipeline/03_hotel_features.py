"""
3단계: 콘텐츠 기반 필터링용 호텔 특징

(1) 호텔 aspect 평균 (n_hotels, 6)
    train 리뷰(완전 + 일부)의 세부 평점을 aspect마다 있는 값만 모아 평균한다.
    리뷰가 적은 호텔의 평균이 튀지 않도록 전체 평균 쪽으로 축소한다 (베이지안 평균).
        a_hk = (Σ 점수 + M · 전체평균_k) / (개수 + M)

(2) 호텔 TF-IDF (n_hotels, vocab)
    train 리뷰의 제목+본문을 호텔별로 모은 단어 빈도에 TF-IDF를 적용한다 (val/test 리뷰 텍스트는 쓰지 않음).
    본문 전체(약 1GB)를 한 번에 올리지 않도록 단어 사전은 표본 10만 건으로 만들고, 빈도는 청크로 누적한다.
    호텔마다 TF-IDF 상위 TOP_TERMS개 단어만 남겨(다시 L2 정규화) 유사도 계산을 가볍게 한다.

출력: data/pipeline/hotel_aspects.npy, hotel_tfidf.npz, vocab.txt
      Pipeline/outputs/03_hotel_top_terms.csv
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from common import ASPECTS, HOTELREC_DIR, SEED, WORK_DIR, ensure_dirs, load_split, save_table

M = 5                 # aspect 평균 축소 강도 (가상 리뷰 수)
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
    """원본 CSV를 청크로 읽어 train 분할에 속한 리뷰의 (hotel_id, 제목+본문)만 돌려준다"""
    for f in ("hotelrec_train.csv", "hotelrec_partial.csv"):
        for ch in pd.read_csv(HOTELREC_DIR / f, usecols=["user_id", "hotel_id", "title", "text"],
                              chunksize=CHUNK, encoding="utf-8-sig"):
            key = ch.user_id.to_numpy(np.int64) * n_hotels + ch.hotel_id.to_numpy(np.int64)
            ch = ch[np.isin(key, train_keys)]
            yield ch.hotel_id.to_numpy(), (ch.title.fillna("") + " " + ch.text.fillna("")).tolist()


def keep_top_terms(T, k):
    """각 행(호텔)에서 값이 큰 k개 단어만 남기고 다시 L2 정규화"""
    T = T.tocsr()
    rows, cols, vals = [], [], []
    for i in range(T.shape[0]):
        s, e = T.indptr[i], T.indptr[i + 1]
        d, idx = T.data[s:e], T.indices[s:e]
        if len(d) > k:
            top = np.argpartition(-d, k)[:k]     # 정렬 없이 상위 k개 위치만 찾기
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

    # (1) aspect 평균
    A = hotel_aspect_means(tr, n_hotels)
    np.save(WORK_DIR / "hotel_aspects.npy", A)
    print("호텔 aspect 평균 저장:", A.shape, "평균", dict(zip(ASPECTS, A.mean(0).round(3))))

    # (2) TF-IDF
    train_keys = np.sort(tr.user_id.to_numpy(np.int64) * n_hotels + tr.hotel_id.to_numpy(np.int64))
    rng = np.random.default_rng(SEED)
    p = VOCAB_SAMPLE / len(tr)
    sample = []
    for _, texts in iter_train_text(train_keys, n_hotels):
        pick = rng.random(len(texts)) < p
        sample += [t for t, m in zip(texts, pick) if m]
    # CountVectorizer: 문서-단어 빈도 행렬을 만든다
    #   stop_words="english": the, and 같은 불용어 제외 / max_features: 빈도 상위 단어만 사전에 포함
    #   min_df=5: 5개 미만 문서에 나온 단어 제외 / token_pattern: 영문자 2글자 이상 단어만 (숫자 제외)
    cv = CountVectorizer(stop_words="english", max_features=MAX_FEATURES, min_df=5,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", dtype=np.float32)
    cv.fit(sample)
    vocab = cv.get_feature_names_out()
    print(f"단어 사전: 표본 {len(sample):,}건 -> {len(vocab):,}개 단어")
    del sample

    C = sp.csr_matrix((n_hotels, len(vocab)), dtype=np.float32)
    n_docs = 0
    for hid, texts in iter_train_text(train_keys, n_hotels):
        X = cv.transform(texts)                                   # (리뷰 수, 단어 수)
        H = sp.csr_matrix((np.ones(len(hid), np.float32), (np.arange(len(hid)), hid)),
                          shape=(len(hid), n_hotels))             # 리뷰 -> 호텔 지시 행렬
        C = C + H.T @ X                                           # 호텔별 단어 빈도 누적
        n_docs += len(hid)
        print(f"   ...{n_docs:,}건 처리")
    # TfidfTransformer: 빈도 행렬에 IDF 가중치를 곱한다
    #   sublinear_tf=True: tf 대신 1+log(tf) (긴 리뷰가 많은 호텔의 영향 완화) / norm="l2": 행 단위 정규화
    T = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(C)
    T = keep_top_terms(T, TOP_TERMS)
    sp.save_npz(WORK_DIR / "hotel_tfidf.npz", T)
    (WORK_DIR / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    print(f"호텔 TF-IDF 저장: {T.shape}, nnz {T.nnz:,}")

    # 리뷰가 많은 호텔 5곳의 상위 단어 (발표/보고서 설명용)
    top_h = tr.hotel_id.value_counts().index[:5]
    rows = []
    for h in top_h:
        r = T.getrow(h)
        order = np.argsort(-r.data)[:10]
        rows.append({"hotel_id": h, "top_terms": ", ".join(vocab[r.indices[order]])})
    top = pd.DataFrame(rows)
    print(top.to_string(index=False))
    save_table(top, "03_hotel_top_terms")


if __name__ == "__main__":
    main()
