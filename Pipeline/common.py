"""
파이프라인 공통 설정과 함수

경로
    data/processing/hotelrec/   전처리 결과 (팀원 Preprocessing/01 출력)
    data/processing/kaggle/     전처리 결과 (팀원 Preprocessing/02 출력)
    data/pipeline/              파이프라인 중간 산출물 (분할, 가중치, 호텔 특징 / Git에 올리지 않음)
    Pipeline/outputs/           결과 표(csv)와 그림(png) (Git에 올림)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOTELREC_DIR = ROOT / "data" / "processing" / "hotelrec"
KAGGLE_DIR = ROOT / "data" / "processing" / "kaggle"
WORK_DIR = ROOT / "data" / "pipeline"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

ASPECTS = ["service", "cleanliness", "value", "rooms", "location", "sleep_quality"]
SEED = 42


def ensure_dirs():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def load_ratings():
    """train(세부 평점 완전) + partial(일부/없음)을 합쳐 평점 데이터만 로드 (본문 제외로 메모리 절약)"""
    cols = ["user_id", "hotel_id", "rating", "date", *ASPECTS]
    parts = [pd.read_csv(HOTELREC_DIR / f, usecols=cols, encoding="utf-8-sig")
             for f in ("hotelrec_train.csv", "hotelrec_partial.csv")]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_split():
    """01_split.py 결과 (split 컬럼: train / val / test)"""
    df = pd.read_csv(WORK_DIR / "ratings_split.csv", parse_dates=["date"])
    n_users, n_hotels = df.user_id.max() + 1, df.hotel_id.max() + 1
    return df, n_users, n_hotels


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def sample_candidates(df_all, users, pos_hotels, n_hotels, n_neg, rng):
    """
    Top-N 평가용 후보 생성: 정답 호텔 1개 + 사용자가 한 번도 평가하지 않은 호텔 n_neg개
    반환: (len(users), 1 + n_neg) 배열, 0번 열이 정답
    """
    rated = df_all.groupby("user_id").hotel_id.apply(set)
    cand = np.empty((len(users), 1 + n_neg), dtype=np.int64)
    cand[:, 0] = pos_hotels
    for i, u in enumerate(users):
        seen = rated[u]
        negs = []
        while len(negs) < n_neg:
            for h in rng.integers(0, n_hotels, size=2 * n_neg):
                if h not in seen and h not in negs:
                    negs.append(h)
                    if len(negs) == n_neg:
                        break
        cand[i, 1:] = negs
    return cand


def ranking_metrics(scores, ks=(5, 10)):
    """
    scores: (사용자 수, 1 + n_neg), 0번 열이 정답 호텔 점수
    정답의 순위 = 정답보다 높은 후보 수 + 동점 후보 수의 절반 (동점이 많은 모델이 유리해지지 않도록)
    HR@k: 정답이 상위 k 안에 들면 1 / NDCG@k: 1/log2(순위+2)
    """
    pos = scores[:, :1]
    rank = (scores[:, 1:] > pos).sum(1) + 0.5 * (scores[:, 1:] == pos).sum(1)
    out = {}
    for k in ks:
        hit = rank < k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1 / np.log2(rank + 2), 0).mean())
    return out, rank


def setup_plot():
    """matplotlib 한글 폰트 설정 (팀원 EDA 노트북과 동일한 방식)"""
    import matplotlib
    matplotlib.use("Agg")                      # 화면 없이 파일로만 저장
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if f in installed:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_fig(plt, name):
    path = OUT_DIR / "figures" / f"{name}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   그림 저장: {path.relative_to(ROOT)}")


def save_table(df, name, index=False):
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"   표 저장: {path.relative_to(ROOT)}")
