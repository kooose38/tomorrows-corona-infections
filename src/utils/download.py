import pandas as pd 

def download_csv():
    URL_1 = "https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv"
    URL_2 = "https://covid19.mhlw.go.jp/public/opendata/requiring_inpatient_care_etc_daily.csv"
    URL_3 = "https://covid19.mhlw.go.jp/public/opendata/deaths_cumulative_daily.csv"
    URL_4 = "https://covid19.mhlw.go.jp/public/opendata/severe_cases_daily.csv"

    target = pd.read_csv(URL_1)
    hos = pd.read_csv(URL_2)
    death = pd.read_csv(URL_3)
    severe = pd.read_csv(URL_4)

    target.to_csv("./data/download/target.csv", index=False)
    hos.to_csv("./data/download/hos.csv", index=False)
    death.to_csv("./data/download/death.csv", index=False)
    severe.to_csv("./data/download/severe.csv", index=False)

    print("sucessfully download csv files")
    