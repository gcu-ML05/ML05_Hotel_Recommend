"""
Step 1: HotelRec train/validation/test split (per-user, time-ordered leave-last-out)

    Over all reviews (train + partial combined), for each user
        the most recent review          -> test
        the review just before it       -> val   (for choosing hyperparameters such as hybrid weights)
        the rest                        -> train (for training all models)
    Preprocessing keeps users with at least 5 reviews, so every user has at least 3 train reviews.
    There is no time leakage (the future is never used to predict the past), and it matches the
    recommendation task of predicting "the next hotel".

    HotelRec dates are mostly monthly (1st of the month), so a user can have reviews on the same date.
    Ties are broken with a fixed-seed random number.

Output: data/pipeline/ratings_split.csv
"""
import numpy as np

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_ratings, save_table


def main():
    ensure_dirs()
    df = load_ratings()
    print(f"total {len(df):,} rows, {df.user_id.nunique():,} users, {df.hotel_id.nunique():,} hotels")

    rng = np.random.default_rng(SEED)
    df["tie"] = rng.random(len(df))
    df = df.sort_values(["user_id", "date", "tie"]).drop(columns="tie")
    # cumcount(ascending=False): numbers each user's reviews 0, 1, 2, ... from the end (0 = most recent)
    from_end = df.groupby("user_id").cumcount(ascending=False).to_numpy()
    df["split"] = np.select([from_end == 0, from_end == 1], ["test", "val"], "train")

    df.to_csv(WORK_DIR / "ratings_split.csv", index=False, date_format="%Y-%m-%d")

    tr = df[df.split == "train"]
    summary = (df.groupby("split")
                 .agg(rows=("rating", "size"), users=("user_id", "nunique"),
                      hotels=("hotel_id", "nunique"), mean_rating=("rating", "mean"))
                 .reindex(["train", "val", "test"]))
    summary["aspects_complete_%"] = (df.groupby("split")[ASPECTS].apply(lambda x: x.notna().all(axis=1).mean() * 100)
                                       .reindex(summary.index).round(1))
    # share of hotels never seen in training (collaborative filtering cannot predict these)
    seen = set(tr.hotel_id)
    summary["unseen_hotel_%"] = [round(100 * (~df[df.split == s].hotel_id.isin(seen)).mean(), 2)
                                 for s in summary.index]
    print(summary.to_string())
    is_last_date = df.date == df.groupby("user_id").date.transform("max")
    print(f"users with 2+ reviews on their latest date (ties broken randomly): "
          f"{(is_last_date.groupby(df.user_id).sum() > 1).mean():.1%}")
    save_table(summary.reset_index(), "01_split_summary")


if __name__ == "__main__":
    main()
