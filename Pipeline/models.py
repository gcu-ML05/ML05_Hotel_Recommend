"""
추천 모델 (제안서 3절의 필터링 방법 + 비교용 기준선)

모든 모델은 같은 인터페이스를 가진다.
    fit(u, i, r, n_users, n_items)   u, i, r: 학습 평점의 사용자 ID / 호텔 ID / 평점 배열
    predict(u, i)                    (u[k], i[k]) 쌍마다 예측 평점(또는 순위 점수) 배열

    BiasBaseline       μ + b_u + b_i                                     (평점 예측 기준선)
    Popularity         호텔 리뷰 수                                        (Top-N 기준선)
    PersonaPopularity  같은 페르소나 사용자들이 높게 평가한 호텔 수           (제안서의 군집 기준선)
    ItemKNN            메모리 기반 CF, 기준 예측 잔차 유사도 + 공동 방문 순위 점수 (제안서 ①)
    BiasedMF           모델 기반 CF, 편향 포함 행렬분해 + L2 + SGD           (제안서 ②)
    ContentBased       aspect 중요도 x 호텔 aspect 평균 + TF-IDF 유사도      (제안서 ③)
    하이브리드          λ·MF + (1-λ)·ContentBased (06_train_evaluate.py)    (제안서 ④)

참고 문헌
    - Sarwar et al. (2001) Item-based collaborative filtering recommendation algorithms. WWW. (adjusted cosine)
    - Koren, Bell, Volinsky (2009) Matrix factorization techniques for recommender systems. IEEE Computer. (biased MF, SGD)
    - Koren (2010) Factor in the neighbors. ACM TKDD. (정규화된 편향 기준선, KNNBaseline)
    - Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS. (공동 방문 유사도 순위)
"""
import time

import numpy as np
import scipy.sparse as sp

from common import SEED, rmse


def user_batches(u, size):
    """
    (u, i) 쌍을 사용자 묶음 단위로 처리하기 위한 도우미.
    yield (users, sel, local): 이 묶음의 고유 사용자 배열, 원래 배열에서의 위치, users 안에서의 위치
    """
    order = np.argsort(u, kind="stable")
    us = u[order]
    uniq, start = np.unique(us, return_index=True)
    start = np.r_[start, len(us)]
    for s in range(0, len(uniq), size):
        e = min(s + size, len(uniq))
        lo, hi = start[s], start[e]
        users = uniq[s:e]
        yield users, order[lo:hi], np.searchsorted(users, us[lo:hi])


class BiasBaseline:
    """r̂ = μ + b_u + b_i. 편향은 정규화된 평균을 번갈아 계산 (Koren 2010)"""

    def __init__(self, reg_u=5.0, reg_i=10.0, n_iter=10):
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter

    def fit(self, u, i, r, n_users, n_items):
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        cu, ci = np.bincount(u, minlength=n_users), np.bincount(i, minlength=n_items)
        for _ in range(self.n_iter):
            self.bi = np.bincount(i, r - self.mu - self.bu[u], n_items) / (ci + self.reg_i)
            self.bu = np.bincount(u, r - self.mu - self.bi[i], n_users) / (cu + self.reg_u)
        return self

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i], 1, 5)


class Popularity:
    """학습 데이터의 리뷰 수가 많은 호텔일수록 높은 점수 (순위 평가 전용)"""

    def fit(self, u, i, r, n_users, n_items):
        self.cnt = np.bincount(i, minlength=n_items).astype(float)
        return self

    def predict(self, u, i):
        return self.cnt[i]


class PersonaPopularity:
    """사용자와 같은 페르소나에 속한 사용자들이 4점 이상 준 횟수가 많은 호텔일수록 높은 점수"""

    def __init__(self, persona):
        self.persona = persona            # (n_users,) 사용자별 페르소나 번호

    def fit(self, u, i, r, n_users, n_items):
        like = r >= 4
        self.M = np.zeros((self.persona.max() + 1, n_items))
        np.add.at(self.M, (self.persona[u[like]], i[like]), 1)
        # 동점일 때는 전체 인기도로 순서를 정하도록 아주 작은 값을 더함
        self.M += 1e-6 * np.bincount(i, minlength=n_items)
        return self

    def predict(self, u, i):
        return self.M[self.persona[u], i]


def _top_k_rows(M, row_offset, k):
    """밀집 행렬 M의 행마다 값이 큰 k개(양수만)를 (행, 열, 값)으로 반환"""
    # np.argpartition: 전체 정렬 없이 상위 k개의 위치만 찾음 (순서는 보장하지 않음)
    top = np.argpartition(-M, k, axis=1)[:, :k]
    v = np.take_along_axis(M, top, axis=1)
    keep = v > 0
    return np.repeat(np.arange(row_offset, row_offset + len(M)), k)[keep.ravel()], top[keep], v[keep]


class ItemKNN:
    """
    메모리 기반 CF (Item-kNN)

    평점 예측 (predict)
        b_ui = 기준 예측. center="baseline"이면 μ + b_u + b_i (KNNBaseline, Koren 2010),
               center="user_mean"이면 사용자 평균 μ_u (adjusted cosine, Sarwar et al. 2001)
        sim(i, j) = 잔차 (r - b) 벡터의 코사인 유사도 x n_ij / (n_ij + shrink)
                    (n_ij: 함께 평가한 사용자 수. 적을수록 유사도를 줄임)
        호텔마다 유사도 상위 k개(양수)만 이웃으로 보관한다.
        r̂(u, h) = b_uh + Σ_{j∈N(h)∩I_u} sim(h,j)(r_uj - b_uj) / (Σ_{j∈N(h)∩I_u} sim(h,j) + damp)
            damp: 사용자가 평가한 이웃이 1~2개뿐일 때 그 잔차를 그대로 믿지 않도록 분모를 키움
            이웃 중 사용자가 평가한 호텔이 없으면 r̂ = b_uh

    Top-N 점수 (rank_score)
        평점 예측은 후보 대부분이 이웃과 겹치지 않아 b_uh로 동점이 되므로 순위용 점수를 따로 둔다.
        score(u, h) = Σ_{j∈I_u} cos_co(h, j)   (사용자가 방문한 호텔들과의 공동 방문 유사도 합)
            cos_co(i, j) = n_ij / sqrt(n_i · n_j)   호텔마다 상위 k개만 보관
        동점은 b_uh로 정한다 (1e-3 배를 더함).
        참고: Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS.
    """

    def __init__(self, k=2000, shrink=100.0, damp=0.03, center="baseline", block=500, user_batch=500):
        self.k, self.shrink, self.damp, self.center = k, shrink, damp, center
        self.block, self.user_batch = block, user_batch

    def _base(self, u, i):
        if self.center == "baseline":
            return self.base.mu + self.base.bu[u] + self.base.bi[i]
        return self.mu_u[u]

    def fit(self, u, i, r, n_users, n_items):
        t0 = time.time()
        if self.center == "baseline":
            self.base = BiasBaseline().fit(u, i, r, n_users, n_items)
        else:
            cnt = np.bincount(u, minlength=n_users)
            self.mu_u = np.bincount(u, r, n_users) / np.maximum(cnt, 1)
        res = (r - self._base(u, i)).astype(np.float32)
        # scipy.sparse.csr_matrix((값, (행, 열)), shape): 0이 대부분인 행렬을 0이 아닌 값만 저장 (CSR 형식)
        self.X = sp.csr_matrix((res, (u, i)), shape=(n_users, n_items), dtype=np.float32)
        self.B = sp.csr_matrix((np.ones_like(res), (u, i)), shape=(n_users, n_items), dtype=np.float32)
        Xt, Bt = self.X.T.tocsr(), self.B.T.tocsr()
        norm = np.sqrt(np.asarray(self.X.multiply(self.X).sum(0)).ravel())
        n_item = np.asarray(self.B.sum(0)).ravel()                  # 호텔별 평가 수 n_i

        S_parts, C_parts = [], []
        kk = min(self.k, n_items - 1)
        # 호텔 x 호텔 유사도 전체(약 4.7만 x 4.7만)는 메모리에 다 올릴 수 없으므로 block개 행씩 계산
        for a in range(0, n_items, self.block):
            b = min(a + self.block, n_items)
            diag = (np.arange(b - a), np.arange(a, b))
            num = (Xt[a:b] @ self.X).toarray()                      # 분자: Σ_u x_ui x_uj
            co = (Bt[a:b] @ self.B).toarray()                       # 함께 평가한 사용자 수 n_ij
            sim = num / (norm[a:b, None] * norm[None, :] + 1e-8) * (co / (co + self.shrink))
            sim[diag] = 0                                           # 자기 자신 제외
            S_parts.append(_top_k_rows(sim, a, kk))
            cos_co = co / (np.sqrt(n_item[a:b, None] * n_item[None, :]) + 1e-8)
            cos_co[diag] = 0
            C_parts.append(_top_k_rows(cos_co, a, kk))

        def to_csr(parts):
            rows, cols, vals = (np.concatenate(x) for x in zip(*parts))
            return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

        S, C = to_csr(S_parts), to_csr(C_parts)
        self.ST, self.CT = S.T.tocsr(), C.T.tocsr()
        print(f"   ItemKNN({self.center}) 유사도 계산 {time.time() - t0:.0f}초, 이웃 수 평균 {S.nnz / n_items:.1f}")
        return self

    def predict(self, u, i):
        out = np.empty(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            # (X_u Sᵀ)[h] = Σ_j x_uj sim(h, j): 사용자가 평가한 이웃 호텔들의 가중합
            num = (self.X[users] @ self.ST).toarray()
            den = (self.B[users] @ self.ST).toarray()
            n, d = num[local, i[sel]], den[local, i[sel]]
            # np.divide(..., where=조건): 조건이 참인 칸만 나누고 나머지는 out 초기값(0) 유지 (0으로 나누기 방지)
            out[sel] = np.divide(n, d + self.damp, out=np.zeros_like(n, dtype=float), where=d > 0)
        return np.clip(self._base(u, i) + out, 1, 5)

    def rank_score(self, u, i):
        out = np.empty(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            out[sel] = (self.B[users] @ self.CT).toarray()[local, i[sel]]
        return out + 1e-3 * self._base(u, i)


class BiasedMF:
    """
    모델 기반 CF: 편향 포함 행렬분해
        r̂(u, i) = μ + b_u + b_i + p_u · q_i
        손실 = Σ (r - r̂)^2 + reg (b_u^2 + b_i^2 + |p_u|^2 + |q_i|^2)     (MSE + L2 정규화)
    SGD로 학습한다. 파이썬 반복문으로 한 건씩 갱신하면 너무 느려서 batch건을 한 번에 갱신하는
    미니배치 SGD를 쓴다 (batch=1이면 수업의 SGD와 같음).
    매 epoch마다 val RMSE를 재고, patience번 연속 나빠지면 멈추고 가장 좋았던 epoch의 파라미터를 쓴다.
    """

    def __init__(self, k=32, lr=0.01, reg=0.05, epochs=40, batch=256, patience=3, seed=SEED):
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        self.batch, self.patience, self.seed = batch, patience, seed

    def _raw(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)

    def fit(self, u, i, r, n_users, n_items, val=None):
        rng = np.random.default_rng(self.seed)
        self.mu = r.mean()
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        lr, reg = self.lr, self.reg
        self.history, best, bad = [], (np.inf, None), 0
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            perm = rng.permutation(len(r))
            for s in range(0, len(r), self.batch):
                idx = perm[s:s + self.batch]
                uu, ii = u[idx], i[idx]
                pu, qi = self.P[uu], self.Q[ii]
                e = r[idx] - (self.mu + self.bu[uu] + self.bi[ii] + (pu * qi).sum(1))   # 오차
                # 손실의 기울기 방향으로 한 걸음: θ ← θ + lr (e · ∂r̂/∂θ - reg · θ)
                np.add.at(self.bu, uu, lr * (e - reg * self.bu[uu]))
                np.add.at(self.bi, ii, lr * (e - reg * self.bi[ii]))
                np.add.at(self.P, uu, lr * (e[:, None] * qi - reg * pu))
                np.add.at(self.Q, ii, lr * (e[:, None] * pu - reg * qi))
            h = {"epoch": ep, "train_rmse": rmse(r, self._raw(u, i))}
            if val is not None:
                h["val_rmse"] = rmse(val[2], np.clip(self._raw(val[0], val[1]), 1, 5))
            self.history.append(h)
            print(f"   MF epoch {ep:2d}  train {h['train_rmse']:.4f}"
                  + (f"  val {h['val_rmse']:.4f}" if val is not None else "") + f"  ({time.time() - t0:.0f}초)")
            if val is None:
                continue
            if h["val_rmse"] < best[0]:
                best, bad = (h["val_rmse"], (self.P.copy(), self.Q.copy(), self.bu.copy(), self.bi.copy())), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best[1] is not None:
            self.P, self.Q, self.bu, self.bi = best[1]
            self.best_epoch = min(self.history, key=lambda x: x["val_rmse"])["epoch"]
        return self

    def predict(self, u, i):
        return np.clip(self._raw(u, i), 1, 5)


class ContentBased:
    """
    콘텐츠 기반 필터링
        aspect 효용  util(u, h) = w_u0 + Σ_k w_uk · a_hk
            w_u : 02단계의 사용자 aspect 중요도, a_h : 03단계의 호텔 aspect 평균
            -> "사용자가 중요하게 여기는 aspect에서 점수가 높은 호텔"일수록 높게 예측
        텍스트 유사도  sim(u, h) = cos(사용자 프로필, 호텔 TF-IDF)
            사용자 프로필 = Σ_j (r_uj - μ_u) · TF-IDF_j  (높게 평가한 호텔의 단어는 +, 낮게 평가한 호텔의 단어는 -)
        r̂(u, h) = util(u, h) + γ (sim(u, h) - s̄)      γ, s̄는 val 평점으로 최소제곱 추정 (fit_gamma)
    """

    def __init__(self, W, A, T, user_batch=500):
        self.W, self.A, self.T, self.user_batch = W, A, T.tocsr(), user_batch
        self.gamma, self.s_mean = 0.0, 0.0

    def fit(self, u, i, r, n_users, n_items):
        cnt = np.bincount(u, minlength=n_users)
        mu_u = np.bincount(u, r, n_users) / np.maximum(cnt, 1)
        c = r - mu_u[u]
        # 모든 평점이 같은 사용자는 가중치가 전부 0이 되므로 평가한 호텔을 같은 비중(1)으로 사용
        flat = np.bincount(u, np.abs(c), n_users) == 0
        c = np.where(flat[u], 1.0, c)
        self.R = sp.csr_matrix((c.astype(np.float32), (u, i)), shape=(n_users, n_items))
        return self

    def utility(self, u, i):
        return self.W[u, 0] + (self.W[u, 1:] * self.A[i]).sum(1)

    def text_sim(self, u, i):
        out = np.zeros(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            P = (self.R[users] @ self.T).toarray()          # (묶음 사용자 수, 단어 수) 사용자 프로필
            norm = np.linalg.norm(P, axis=1)
            norm[norm == 0] = 1
            Th = self.T[i[sel]]                             # 대상 호텔들의 TF-IDF (희소)
            row = np.repeat(np.arange(len(sel)), np.diff(Th.indptr))
            # 희소 x 밀집 내적: 호텔 벡터의 0이 아닌 칸만 사용자 프로필에서 꺼내 곱한 뒤 행별로 합산
            dot = np.bincount(row, Th.data * P[local[row], Th.indices], minlength=len(sel))
            out[sel] = dot / norm[local]
        return out

    def fit_gamma(self, u, i, r):
        util, sim = self.utility(u, i), self.text_sim(u, i)
        self.s_mean = sim.mean()
        z = sim - self.s_mean
        self.gamma = float((z * (r - util)).sum() / (z * z).sum())
        print(f"   ContentBased γ = {self.gamma:.3f} (텍스트 유사도 평균 {self.s_mean:.3f})")
        return self

    def predict(self, u, i):
        return np.clip(self.utility(u, i) + self.gamma * (self.text_sim(u, i) - self.s_mean), 1, 5)
