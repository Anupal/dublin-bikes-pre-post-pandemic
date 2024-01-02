from datetime import datetime, timedelta


# YYYY-MM-DD
ALL_PUBLIC_HOLIDAYS = {
    "2018-08-06",
    "2018-10-29",
    "2018-12-25",
    "2018-12-26",
    "2019-01-01",
    "2019-03-17",
    "2019-04-22",
    "2019-05-06",
    "2019-06-03",
    "2019-08-05",
    "2019-10-28",
    "2019-12-25",
    "2019-12-26",
    "2020-01-01",
    "2020-03-17",
    "2020-04-13",
    "2020-05-04",
    "2020-06-01",
    "2020-08-03",
    "2020-10-26",
    "2020-12-25",
    "2020-12-26",
    "2021-01-01",
    "2020-03-17",
    "2020-04-05",
    "2020-05-03",
    "2020-06-07",
    "2020-08-02",
    "2020-10-25",
    "2020-12-25",
    "2020-12-26",
    "2021-01-01",
    "2021-03-17",
    "2021-03-18",
    "2021-04-18",
    "2021-05-02",
    "2021-06-06",
    "2021-08-01",
    "2021-10-31",
    "2021-12-25",
    "2021-12-26",
    "2022-01-01",
    "2022-03-17",
    "2022-03-18",
    "2022-04-18",
    "2022-05-02",
    "2022-06-06",
    "2022-08-01",
    "2022-10-31",
    "2022-12-25",
    "2022-12-26",
    "2023-01-01",
    "2023-02-06",
    "2023-03-17",
    "2023-04-10",
    "2023-05-01",
    "2023-06-05",
    "2023-08-07",
    "2023-10-30",
    "2023-12-25",
    "2023-12-26",
}

def weekends_by_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    weekends = set()

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() in [5, 6]:  # 5 is Saturday, 6 is Sunday
            weekends.add(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return weekends

ALL_WEEKENDS = set()
for year in (2018, 2019, 2020, 2021, 2022, 2023):
    ALL_WEEKENDS.update(weekends_by_year(year))

def check_weekend(date_obj):
    return int(date_obj.strftime('%Y-%m-%d') in ALL_WEEKENDS)

def check_public_holiday(date_obj):
    return int(date_obj.strftime('%Y-%m-%d') in ALL_PUBLIC_HOLIDAYS)