"""
HotelRec(JSON Lines, .txt) -> Hotel_Reviews(Kaggle)와 같은 6개 도시 호텔만 추려서 CSV로 저장

- 파일 전체를 메모리에 올리지 않고 한 줄씩 읽어서 처리 (용량이 커도 OK)
- 한 줄에 JSON 하나씩인 형식(JSON Lines)을 그대로 파싱 -> 컬럼이 깨지지 않음
- 도시 조건에 맞는 행만 JSON 파싱하므로 빠름

사용법:
    python hotelrec_filter_to_csv.py
    (아래 SRC / DST 경로만 본인 환경에 맞게 수정)
"""
import csv
import json
import re

# ---------------- 설정 ----------------
SRC = r"../data/raw/HotelRec.txt"            # 원본 txt (JSON Lines)
DST = r"HotelRec_6cities.csv"             # 결과 csv
KEEP_TEXT = True                          # False면 리뷰 본문(text) 제외 -> 용량 대폭 감소
COLUMNS = ["hotel_url", "hotel_name", "city", "author", "date", "rating", "title", "text"]
# --------------------------------------

# hotel_url 맨 끝의 "지역" 부분이 아래 패턴일 때만 통과
# 예) ...-Reviews-Hotel_Arena-Amsterdam_North_Holland_Province.html
CITY_PATTERNS = {
    "London":    r"London_England",
    "Paris":     r"Paris_Ile_de_France",
    "Barcelona": r"Barcelona_Catalonia",
    "Milan":     r"Milan_Lombardy",
    "Vienna":    r"Vienna",                         # URL에서 지역이 'Vienna' 단독
    "Amsterdam": r"Amsterdam_North_Holland_Province",
}
URL_RE = re.compile(r"Reviews-(?P<name>.+)-(?P<loc>[^-]+)\.html$")
CITY_RES = {c: re.compile(rf"^{p}$") for c, p in CITY_PATTERNS.items()}

# 빠른 1차 필터용 (JSON 파싱 전에 문자열 포함 여부만 확인)
QUICK = ("London_England", "Paris_Ile_de_France", "Barcelona_Catalonia",
         "Milan_Lombardy", "-Vienna.html", "Amsterdam_North_Holland_Province")


def parse_url(url):
    """hotel_url -> (hotel_name, city) / 대상 도시가 아니면 (None, None)"""
    m = URL_RE.search(url)
    if not m:
        return None, None
    loc = m.group("loc")
    for city, rx in CITY_RES.items():
        if rx.match(loc):
            return m.group("name").replace("_", " "), city
    return None, None


def main():
    total = kept = bad = 0
    hotels = set()

    with open(SRC, "r", encoding="utf-8") as fin, \
         open(DST, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for line in fin:
            total += 1
            if not any(q in line for q in QUICK):      # 1차: 문자열로 빠르게 제외
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            name, city = parse_url(rec.get("hotel_url", ""))   # 2차: 정확히 확인
            if city is None:
                continue

            rec["hotel_name"], rec["city"] = name, city
            if not KEEP_TEXT:
                rec["text"] = ""
            writer.writerow(rec)
            kept += 1
            hotels.add(rec["hotel_url"])

            if kept % 100_000 == 0:
                print(f"  ...{kept:,} rows kept (scanned {total:,})")

    print(f"완료: 전체 {total:,}줄 중 {kept:,}행 저장, 호텔 {len(hotels):,}개, 파싱 실패 {bad}줄")
    print(f"저장 위치: {DST}")


if __name__ == "__main__":
    main()