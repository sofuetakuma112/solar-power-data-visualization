import os
import serial
from elasticsearch import Elasticsearch
import datetime
import time
import codecs
import binascii
from dotenv import load_dotenv

load_dotenv(f"{os.getcwd()}/.env")

###ASCIIを16進数コードへ変換
def convert(a):
    byte_date_ascii = a.encode("ASCII")
    byte_date_16 = binascii.b2a_hex(byte_date_ascii)
    date_16_a = byte_date_16.decode("ASCII")
    return date_16_a


###raspi起動時の処理 (raspiの時刻同期を待つ)
time.sleep(60)
es = Elasticsearch(
    "http://133.71.201.197:9200",
    http_auth=(
        os.getenv("RECYCLE_ELASTIC_USER_NAME"),
        os.getenv("RECYCLE_ELASTIC_PASSWORD"),
    ),
)

while True:
    ser = serial.Serial("/dev/ttyUSB0S", 19200)
    # print("ポートopen")
    ###初期化
    date_t = ""
    STX = "false"
    ETX = "false"
    ###電文を検出する繰り返し
    while True:
        try:
            byte_date = ser.read()
            # print(byte_date)
        except:
            print("読み込みエラー")
            ser.close()
            time.sleep(0.1)
            break
        if STX == "false" and ETX == "false" and byte_date == b"\x02":
            # print("02 get")
            STX = "true"
            continue
        elif STX == "true" and ETX == "false" and byte_date == b"\x03":
            # print("03 get")
            ETX = "true"
            continue
        elif STX == "true" and ETX == "true" and byte_date == b"\r":
            # print("cr get")
            ###初期化
            STX = "false"
            ETX = "false"
            ###電文取得後の処理
            if date_t[0:8] == "AA0001Aa":
                ###電文を取得した時間
                dt = datetime.datetime.now()  # JST時間をタイムゾーンをUTCとして取得する
                udt = dt - datetime.timedelta(hours=9)
                print("電文&取得時間get")
                ###チェックサムデータの準備
                try:
                    check_t = date_t[-2:]  ###電文のチェックサム
                    ###チェックサム計算用のデータを電文より作成
                    date_16 = convert(date_t)
                    calc_date = "02" + date_16[:-4] + "03"
                    ###チェックサム計算開始
                    k = 0
                    l = 2
                    d = 0
                    for _ in range(len(calc_date) // 2):
                        f = calc_date[k:l]
                        k += 2
                        l += 2
                        c = int(f, 16)  ###10進数へ
                        d += c  ###加算
                        e = hex(d)  ###16進数
                    ###下位１バイトを抽出して大文字に変換
                    calcChecksum_f = e[-2] + e[-1]
                    calcChecksum_t = calcChecksum_f.upper()
                    # print("チェックサムの比較",check_t,calcChecksum_t)

                except:
                    print("error1")
                    date_t = ""
                    continue

                    ###チェックサムが一致する
                if check_t == calcChecksum_t:
                    # print("チェックtrue",date_t)
                    ###ポートクローズ
                    ser.close()
                    # print("ポートクローズ")
                    time.sleep(0.1)
                    ###不明
                    no_0 = date_t.split(",")[0]
                    no_1 = date_t.split(",")[1]
                    no_2 = date_t.split(",")[2]
                    no_3 = date_t.split(",")[3]
                    no_4 = date_t.split(",")[4]
                    no_5 = date_t.split(",")[5]
                    no_6 = date_t.split(",")[6]
                    no_7 = date_t.split(",")[7]
                    ###直流電圧(V)
                    try:
                        dc_voltage = float(date_t.split(",")[8]) * 0.1
                        DC_Voltage = "{:.3f}".format(dc_voltage)
                        A = float(DC_Voltage)
                    except:
                        A = None
                    ###直流電流(A)
                    try:
                        dc = float(date_t.split(",")[9]) * 0.01
                        DC = "{:.3f}".format(dc)
                        B = float(DC)
                    except:
                        B = None
                    ###直流電力(kw)
                    try:
                        dc_power = float(date_t.split(",")[10]) * 0.01
                        DC_Power = "{:.3f}".format(dc_power)
                        C = float(DC_Power)
                    except:
                        C = None
                    ###交流電圧(V)
                    try:
                        ac_voltage = float(date_t.split(",")[11]) * 0.1
                        AC_Voltage = "{:.3f}".format(ac_voltage)
                        D = float(AC_Voltage)
                    except:
                        D = None
                    ###交流電流 (A)
                    try:
                        ac = float(date_t.split(",")[12]) * 0.01
                        AC = "{:.3f}".format(ac)
                        E = float(AC)
                    except:
                        E = None
                    ###交流電力(kw)
                    try:
                        ac_power = float(date_t.split(",")[13]) * 0.01
                        AC_Power = "{:.3f}".format(ac_power)
                        F = float(AC_Power)
                    except:
                        F = None
                    ###周波数(Hz)
                    try:
                        frequency1 = float(date_t.split(",")[14]) * 0.1
                        frequency2 = "{:.3f}".format(frequency1)
                        G = float(frequency2)
                    except:
                        G = None
                    ###単機積算発電量(kwh)single_unit_integrated_power_generation
                    H = float(date_t.split(",")[15])
                    ###不明
                    no_16 = date_t.split(",")[16]
                    ###総合交流電力(正)(kw)
                    try:
                        total_ac_power1 = float(date_t.split(",")[17]) * 0.01
                        total_ac_power2 = "{:.3f}".format(total_ac_power1)
                        I = float(total_ac_power2)
                    except:
                        I = None
                    ###不明
                    no_18 = date_t.split(",")[18]
                    ###総合積算発電量(kwh)total_unit_integrated_power_generation
                    J = float(date_t.split(",")[19])
                    ###不明
                    no_20 = date_t.split(",")[20]
                    no_21 = date_t.split(",")[21]
                    ###太陽電池電流(A)
                    try:
                        solar_cell_current1 = float(date_t.split(",")[22]) * 0.01
                        solar_cell_current2 = "{:.3f}".format(solar_cell_current1)
                        K = float(solar_cell_current2)
                    except:
                        K = None
                    ###太陽電池電力(kw)
                    try:
                        solar_cell_power1 = float(date_t.split(",")[23]) * 0.01
                        solar_cell_power2 = "{:.3f}".format(solar_cell_power1)
                        L = float(solar_cell_power2)
                    except:
                        L = None
                    ###太陽電池電圧(V)
                    try:
                        solar_cell_voltage1 = float(date_t.split(",")[24]) * 0.1
                        solar_cell_voltage2 = "{:.3f}".format(solar_cell_voltage1)
                        M = float(solar_cell_voltage2)
                    except:
                        M = None
                    ###不明
                    no_25 = date_t.split(",")[25]
                    no_26 = date_t.split(",")[26]
                    ###蓄電池残容量(%)
                    N = float(date_t.split(",")[27])
                    ###日射強度(kW/㎡)へ変換
                    try:
                        solarIrradiance_f = float(date_t.split(",")[28]) * 0.001
                        solarIrradiance = (
                            0.356885525872923 * solarIrradiance_f - 0.356534291543566
                        )
                        SolarIrradiance = "{:.3f}".format(solarIrradiance)
                        O = float(SolarIrradiance)
                    except:
                        O = None
                    ###気温(℃)へ変換
                    try:
                        airTemperature_f = float(date_t.split(",")[29]) * 0.001
                        airTemperature = 25 * airTemperature_f - 75
                        AirTemperature = "{:.3f}".format(airTemperature)
                        P = float(AirTemperature)
                    except:
                        P = None
                    ###不明
                    no_30 = date_t.split(",")[30]
                    no_31 = date_t.split(",")[31]
                    no_32 = date_t.split(",")[32]
                    ###二酸化炭素削減量(kg-CO2)
                    try:
                        co2_reduction1 = F * 0.5335
                        co2_reduction2 = "{:.3f}".format(co2_reduction1)
                        Q = float(co2_reduction2)
                    except:
                        Q = None
                    ###原油換算量(L)
                    try:
                        oil_conversion_amount1 = F * 0.227
                        oil_conversion_amount2 = "{:.3f}".format(oil_conversion_amount1)
                        R = float(oil_conversion_amount2)
                    except:
                        R = None

                    ###Elasticsearchへデータ投入
                    try:
                        es.index(
                            index="pcs_test4",
                            document={
                                "utctime": udt,
                                "JPtime": dt,
                                "NO_0": no_0,
                                "NO_1": no_1,
                                "NO_2": no_2,
                                "NO_3": no_3,
                                "NO_4": no_4,
                                "NO_5": no_5,
                                "NO_6": no_6,
                                "NO_7": no_7,
                                "dc-v(V)": A,
                                "dc-i(A)": B,
                                "dc-pw(kw)": C,
                                "ac-v(V)": D,
                                "ac-i(A)": E,
                                "ac-pw(kw)": F,
                                "frequency(Hz)": G,
                                "single_unit_integrated_power_generation(kwh)": H,
                                "NO_16": no_16,
                                "total_ac_power(kw)": I,
                                "NO_18": no_18,
                                "total_unit_integrated_power_generation(kwh)": J,
                                "NO_20": no_20,
                                "NO_21": no_21,
                                "solar_cell_current(A)": K,
                                "solar_cell_power(kw)": L,
                                "solar_cell_voltage(V)": M,
                                "NO_25": no_25,
                                "NO_26": no_26,
                                "remaining storage battery capacity(%)": N,
                                "solarIrradiance(kw/m^2)": O,
                                "airTemperature(℃)": P,
                                "NO_30": no_30,
                                "NO_31": no_31,
                                "NO_32": no_32,
                                "co2_reduction(kg-CO2)": Q,
                                "oil_conversion_amount(L)": R,
                            },
                        )
                        date_t = ""
                        # print("データ投入成功")
                        break
                    except:
                        print("データ投入失敗")
                        date_t = ""
                        break

                ###チェックサムが一致しない
                else:
                    date_t = ""
                    continue

            ###文頭が"AA0001Aa"以外
            else:
                date_t = ""
                continue

        elif STX == "false" and ETX == "false":
            continue
        elif STX == "true" and ETX == "false":
            try:
                date = byte_date.decode("ASCII")
                date_t += date
                continue
            except:
                print("デコードエラー")
                STX = "false"
                ETX = "false"
                date_t = ""
                continue
        elif STX == "true" and ETX == "true":
            try:
                date = byte_date.decode("ASCII")
                date_t += date
                continue
            except:
                print("デコードエラー")
                STX = "false"
                ETX = "false"
                date_t = ""
                continue
