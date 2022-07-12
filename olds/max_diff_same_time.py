maxDiffAndIndex = [0, 0, None, None]
diffs_q_and_q_calc = []
for i in range(len(qs_calc_scaled)):
    diff = np.abs(qs[i] - qs_calc_scaled[i])
    diffs_q_and_q_calc.append(diff)
    if diff > maxDiffAndIndex[0]:
        maxDiffAndIndex[0] = diff
        maxDiffAndIndex[1] = i
        maxDiffAndIndex[2] = qs[i]
        maxDiffAndIndex[3] = qs_calc_scaled[i]

max_diff = maxDiffAndIndex[0]
max_diff_idx = maxDiffAndIndex[1]
max_diff_q = maxDiffAndIndex[2]
max_diff_q_calc = maxDiffAndIndex[3]

print(max_diff, max_diff_idx, max_diff_q, max_diff_q_calc)

# plt.hlines(max_diff_q_calc, dtsByMinute[0], dtsByMinute[-1])
# plt.vlines(dtsByMinute[max_diff_idx], 0, max(qs_calc_scaled))

# plt.vlines(dtsByMinute[same_q_index], 0, max(qs_calc_scaled))

# dt_calc = datetime.datetime(
#     times[0].year, times[0].month, times[0].day, 0, 0, 0, 0
# ) + datetime.timedelta(minutes=same_q_index)
# dt_q = datetime.datetime(
#     times[0].year, times[0].month, times[0].day, 0, 0, 0, 0
# ) + datetime.timedelta(minutes=max_diff_idx)

# print(f"time diff: {dt_q - dt_calc}")

# 実測値を先頭からループで見て、差がmax_diff以上になるインデックス距離の平均を求める
# left_index = 0
# minite_diffs = []
# for i in range(len(qs_calc_scaled)):
#     diff = np.abs(qs[i] - qs[left_index])
#     if diff >= max_diff:
#         dt_left = datetime.datetime(
#             times[0].year, times[0].month, times[0].day, 0, 0, 0, 0
#         ) + datetime.timedelta(minutes=left_index)
#         dt_right = datetime.datetime(
#             times[0].year, times[0].month, times[0].day, 0, 0, 0, 0
#         ) + datetime.timedelta(minutes=i)
#         print(dt_left, dt_right, np.abs(i - left_index))
#         minite_diffs.append(np.abs(i - left_index))
#         left_index = i + 1

# print(f"mean diff minutes: {sum(minite_diffs[1:])/len(minite_diffs[1:])}")

# plt.plot(  # 理論値をプロット
#     dtsByMinute,
#     diffs_q_and_q_calc,  # 縦軸のスケールを実測値と揃えている
#     label="差分",
# )
