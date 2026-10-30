"""
1단계: HotelRec 학습/검증/테스트 분할 (사용자별 시간 순 leave-last-out)

    train + partial을 합친 전체 리뷰에서 사용자마다
        가장 최근 리뷰 1건   -> test
        그 직전 리뷰 1건     -> val   (하이브리드 가중치 등 하이퍼파라미터 선택용)
        나머지              -> train (모든 모델 학습용)
    전처리에서 사용자당 리뷰가 5건 이상이므로 train에는 사용자마다 최소 3건이 남는다.
    미래 리뷰로 과거를 맞히는 시간 누수가 없고, "다음에 갈 호텔"을 맞히는 추천 상황과 같다.

    HotelRec 날짜는 대부분 월 단위(매월 1일)라 같은 날짜의 리뷰가 있을 수 있다.
    동률은 시드 고정 난수로 순서를 정한다.

출력: data/pipeline/ratings_split.csv
"""
import numpy as np

from common import ASPECTS, SEED, WORK_DIR, ensure_dirs, load_ratings, save_table


def main():
    ensure_dirs()
    df = load_ratings()
    print(f"전체 {len(df):,}행, 사용자 {df.user_id.nunique():,}명, 호텔 {df.hotel_id.nunique():,}개")

    rng = np.random.default_rng(SEED)
    df["tie"] = rng.random(len(df))
    df = df.sort_values(["user_id", "date", "tie"]).drop(columns="tie")
    # cumcount(ascending=False): 사용자별로 뒤에서부터 0, 1, 2, ... 번호 (0 = 가장 최근)
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
    # 학습 데이터에 한 번도 나오지 않은 호텔(협업 필터링이 예측할 수 없는 경우)의 비율
    seen = set(tr.hotel_id)
    summary["unseen_hotel_%"] = [round(100 * (~df[df.split == s].hotel_id.isin(seen)).mean(), 2)
                                 for s in summary.index]
    print(summary.to_string())
    is_last_date = df.date == df.groupby("user_id").date.transform("max")
    print(f"가장 최근 날짜에 리뷰가 2건 이상인 사용자 비율(동률 처리 대상): "
          f"{(is_last_date.groupby(df.user_id).sum() > 1).mean():.1%}")
    save_table(summary.reset_index(), "01_split_summary")


if __name__ == "__main__":
    main()
