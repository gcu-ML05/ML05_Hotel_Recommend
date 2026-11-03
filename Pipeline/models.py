"""
Recommendation models (the filtering methods in section 3 of the proposal + baselines for comparison)

All models share the same interface.
    fit(u, i, r, n_users, n_items)   u, i, r: user ID / hotel ID / rating arrays of the training ratings
    predict(u, i)                    array of predicted ratings (or ranking scores) for each (u[k], i[k]) pair

    BiasBaseline       μ + b_u + b_i                                     (rating prediction baseline)
    Popularity         number of hotel reviews                           (Top-N baseline)
    PersonaPopularity  count of high ratings by users of the same persona (the proposal's cluster baseline)
    ItemKNN            memory-based CF, baseline-residual similarity + co-visit ranking score (proposal ①)
    BiasedMF           model-based CF, biased matrix factorization + L2 + SGD (proposal ②)
    ContentBased       aspect importance x hotel aspect means + TF-IDF similarity (proposal ③)
    Hybrid             λ·MF + (1-λ)·ContentBased (06_train_evaluate.py)  (proposal ④)

References
    - Sarwar et al. (2001) Item-based collaborative filtering recommendation algorithms. WWW. (adjusted cosine)
    - Koren, Bell, Volinsky (2009) Matrix factorization techniques for recommender systems. IEEE Computer. (biased MF, SGD)
    - Koren (2010) Factor in the neighbors. ACM TKDD. (regularized bias baseline, KNNBaseline)
    - Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS. (co-visit similarity ranking)
"""
import time

import numpy as np
import scipy.sparse as sp

from common import SEED, rmse


def user_batches(u, size):
    """
    Helper for processing (u, i) pairs in batches of users.
    yield (users, sel, local): unique users of this batch, positions in the original arrays, positions within users
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
    """r̂ = μ + b_u + b_i. The biases are computed as alternating regularized means (Koren 2010)"""

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
    """Hotels with more reviews in the training data score higher (ranking evaluation only)"""

    def fit(self, u, i, r, n_users, n_items):
        self.cnt = np.bincount(i, minlength=n_items).astype(float)
        return self

    def predict(self, u, i):
        return self.cnt[i]


class PersonaPopularity:
    """Hotels that users of the same persona as the user rated 4 or higher more often score higher"""

    def __init__(self, persona):
        self.persona = persona            # (n_users,) persona index of each user

    def fit(self, u, i, r, n_users, n_items):
        like = r >= 4
        self.M = np.zeros((self.persona.max() + 1, n_items))
        np.add.at(self.M, (self.persona[u[like]], i[like]), 1)
        # add a tiny amount so that ties are broken by overall popularity
        self.M += 1e-6 * np.bincount(i, minlength=n_items)
        return self

    def predict(self, u, i):
        return self.M[self.persona[u], i]


def _top_k_rows(M, row_offset, k):
    """For each row of the dense matrix M, return the k largest (positive only) values as (row, col, value)"""
    # np.argpartition: find only the positions of the top k without a full sort (order not guaranteed)
    top = np.argpartition(-M, k, axis=1)[:, :k]
    v = np.take_along_axis(M, top, axis=1)
    keep = v > 0
    return np.repeat(np.arange(row_offset, row_offset + len(M)), k)[keep.ravel()], top[keep], v[keep]


class ItemKNN:
    """
    Memory-based CF (Item-kNN)

    Rating prediction (predict)
        b_ui = baseline prediction. center="baseline": μ + b_u + b_i (KNNBaseline, Koren 2010),
               center="user_mean": user mean μ_u (adjusted cosine, Sarwar et al. 2001)
        sim(i, j) = cosine similarity of the residual (r - b) vectors x n_ij / (n_ij + shrink)
                    (n_ij: number of users who rated both. The fewer, the more the similarity is shrunk)
        Only the top k (positive) neighbors by similarity are kept per hotel.
        r̂(u, h) = b_uh + Σ_{j∈N(h)∩I_u} sim(h,j)(r_uj - b_uj) / (Σ_{j∈N(h)∩I_u} sim(h,j) + damp)
            damp: enlarges the denominator so the residuals are not trusted as is when the user rated only 1~2 neighbors
            if the user rated none of the neighbors, r̂ = b_uh

    Top-N score (rank_score)
        In rating prediction most candidates share no neighbors with the user and tie at b_uh, so a separate ranking score is used.
        score(u, h) = Σ_{j∈I_u} cos_co(h, j)   (sum of co-visit similarities to the hotels the user visited)
            cos_co(i, j) = n_ij / sqrt(n_i · n_j)   only the top k kept per hotel
        Ties are broken by b_uh (adding 1e-3 times it).
        Reference: Deshpande & Karypis (2004) Item-based top-N recommendation algorithms. ACM TOIS.
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
        # scipy.sparse.csr_matrix((values, (rows, cols)), shape): store a mostly-zero matrix by its nonzero values only (CSR format)
        self.X = sp.csr_matrix((res, (u, i)), shape=(n_users, n_items), dtype=np.float32)
        self.B = sp.csr_matrix((np.ones_like(res), (u, i)), shape=(n_users, n_items), dtype=np.float32)
        Xt, Bt = self.X.T.tocsr(), self.B.T.tocsr()
        norm = np.sqrt(np.asarray(self.X.multiply(self.X).sum(0)).ravel())
        n_item = np.asarray(self.B.sum(0)).ravel()                  # number of ratings per hotel n_i

        S_parts, C_parts = [], []
        kk = min(self.k, n_items - 1)
        # the full hotel x hotel similarity (about 47k x 47k) does not fit in memory, so compute block rows at a time
        for a in range(0, n_items, self.block):
            b = min(a + self.block, n_items)
            diag = (np.arange(b - a), np.arange(a, b))
            num = (Xt[a:b] @ self.X).toarray()                      # numerator: Σ_u x_ui x_uj
            co = (Bt[a:b] @ self.B).toarray()                       # number of users who rated both, n_ij
            sim = num / (norm[a:b, None] * norm[None, :] + 1e-8) * (co / (co + self.shrink))
            sim[diag] = 0                                           # exclude itself
            S_parts.append(_top_k_rows(sim, a, kk))
            cos_co = co / (np.sqrt(n_item[a:b, None] * n_item[None, :]) + 1e-8)
            cos_co[diag] = 0
            C_parts.append(_top_k_rows(cos_co, a, kk))

        def to_csr(parts):
            rows, cols, vals = (np.concatenate(x) for x in zip(*parts))
            return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

        S, C = to_csr(S_parts), to_csr(C_parts)
        self.ST, self.CT = S.T.tocsr(), C.T.tocsr()
        print(f"   ItemKNN({self.center}) similarity computed in {time.time() - t0:.0f}s, mean neighbors {S.nnz / n_items:.1f}")
        return self

    def predict(self, u, i):
        out = np.empty(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            # (X_u Sᵀ)[h] = Σ_j x_uj sim(h, j): weighted sum over the neighbor hotels the user rated
            num = (self.X[users] @ self.ST).toarray()
            den = (self.B[users] @ self.ST).toarray()
            n, d = num[local, i[sel]], den[local, i[sel]]
            # np.divide(..., where=cond): divide only where cond is true and keep the initial out value (0) elsewhere (avoids division by zero)
            out[sel] = np.divide(n, d + self.damp, out=np.zeros_like(n, dtype=float), where=d > 0)
        return np.clip(self._base(u, i) + out, 1, 5)

    def rank_score(self, u, i):
        out = np.empty(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            out[sel] = (self.B[users] @ self.CT).toarray()[local, i[sel]]
        return out + 1e-3 * self._base(u, i)


class BiasedMF:
    """
    Model-based CF: biased matrix factorization
        r̂(u, i) = μ + b_u + b_i + p_u · q_i
        loss = Σ (r - r̂)^2 + reg (b_u^2 + b_i^2 + |p_u|^2 + |q_i|^2)     (MSE + L2 regularization)
    Trained with SGD. Updating one rating at a time in a Python loop is too slow, so mini-batch SGD
    updates batch ratings at once (batch=1 is the same as the SGD from class).
    val RMSE is measured every epoch; after patience consecutive worse epochs training stops and the best epoch's parameters are used.
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
                e = r[idx] - (self.mu + self.bu[uu] + self.bi[ii] + (pu * qi).sum(1))   # error
                # one step along the loss gradient: θ ← θ + lr (e · ∂r̂/∂θ - reg · θ)
                np.add.at(self.bu, uu, lr * (e - reg * self.bu[uu]))
                np.add.at(self.bi, ii, lr * (e - reg * self.bi[ii]))
                np.add.at(self.P, uu, lr * (e[:, None] * qi - reg * pu))
                np.add.at(self.Q, ii, lr * (e[:, None] * pu - reg * qi))
            h = {"epoch": ep, "train_rmse": rmse(r, self._raw(u, i))}
            if val is not None:
                h["val_rmse"] = rmse(val[2], np.clip(self._raw(val[0], val[1]), 1, 5))
            self.history.append(h)
            print(f"   MF epoch {ep:2d}  train {h['train_rmse']:.4f}"
                  + (f"  val {h['val_rmse']:.4f}" if val is not None else "") + f"  ({time.time() - t0:.0f}s)")
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
    Content-based filtering
        aspect utility  util(u, h) = w_u0 + Σ_k w_uk · a_hk
            w_u : user aspect importance from step 02, a_h : hotel aspect means from step 03
            -> the higher a hotel scores on the aspects the user cares about, the higher the prediction
        text similarity  sim(u, h) = cos(user profile, hotel TF-IDF)
            user profile = Σ_j (r_uj - μ_u) · TF-IDF_j  (words of highly rated hotels +, words of low rated hotels -)
        r̂(u, h) = util(u, h) + γ (sim(u, h) - s̄)      γ, s̄ estimated by least squares on the val ratings (fit_gamma)
    """

    def __init__(self, W, A, T, user_batch=500):
        self.W, self.A, self.T, self.user_batch = W, A, T.tocsr(), user_batch
        self.gamma, self.s_mean = 0.0, 0.0

    def fit(self, u, i, r, n_users, n_items):
        cnt = np.bincount(u, minlength=n_users)
        mu_u = np.bincount(u, r, n_users) / np.maximum(cnt, 1)
        c = r - mu_u[u]
        # a user whose ratings are all equal would get all-zero weights, so their rated hotels get equal weight (1)
        flat = np.bincount(u, np.abs(c), n_users) == 0
        c = np.where(flat[u], 1.0, c)
        self.R = sp.csr_matrix((c.astype(np.float32), (u, i)), shape=(n_users, n_items))
        return self

    def utility(self, u, i):
        return self.W[u, 0] + (self.W[u, 1:] * self.A[i]).sum(1)

    def text_sim(self, u, i):
        out = np.zeros(len(u))
        for users, sel, local in user_batches(u, self.user_batch):
            P = (self.R[users] @ self.T).toarray()          # (users in batch, terms) user profiles
            norm = np.linalg.norm(P, axis=1)
            norm[norm == 0] = 1
            Th = self.T[i[sel]]                             # TF-IDF of the target hotels (sparse)
            row = np.repeat(np.arange(len(sel)), np.diff(Th.indptr))
            # sparse x dense dot product: take only the hotel vector's nonzero entries from the user profile, multiply and sum per row
            dot = np.bincount(row, Th.data * P[local[row], Th.indices], minlength=len(sel))
            out[sel] = dot / norm[local]
        return out

    def fit_gamma(self, u, i, r):
        util, sim = self.utility(u, i), self.text_sim(u, i)
        self.s_mean = sim.mean()
        z = sim - self.s_mean
        self.gamma = float((z * (r - util)).sum() / (z * z).sum())
        print(f"   ContentBased γ = {self.gamma:.3f} (mean text similarity {self.s_mean:.3f})")
        return self

    def predict(self, u, i):
        return np.clip(self.utility(u, i) + self.gamma * (self.text_sim(u, i) - self.s_mean), 1, 5)
