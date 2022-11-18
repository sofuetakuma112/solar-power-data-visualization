import pandas as pd
import codecs
import datetime

filepath = "data/DougoSyou/1809010000.csv"
with codecs.open(filepath, "r", "Shift-JIS", "ignore") as file:
    df = pd.read_csv(file, delimiter=",", skiprows=[1])
    # print(df)

    for index, row in df.iterrows():
        print(f"TIME: {row['TIME']}")
        year = int(row['TIME'][:4])
        month = int(row['TIME'][5:7])
        day = int(row['TIME'][8:10])
        hour = int(row['TIME'][11:13])
        minute = int(row['TIME'][14:])
        print(f"dateTime: {datetime.datetime(year, month, day, hour, minute)}") # JPtime?
        print(f"日射強度: {row['日射強度']}") # elasticsearchのsolarIrradiance(kw/m^2)に対応？
        print(f"外気温度: {row['外気温度']}") # airTemperature(℃)
        print(f"直流電力: {row['直流電力']}") # dc-pw(kw)
        print(f"交流電力: {row['交流電力']}") # ac-pw(kw)?
        break
