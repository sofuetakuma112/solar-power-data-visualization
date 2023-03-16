import numpy as np


def calc_delay(a, b):
    # aを固定して、bを左から右へスライドさせていく
    corr = np.correlate(a, b, "full")

    # estimated_delay が負の場合、
    # a に対して b を進ませた場合に相互相関が最大となったと言えます。
    # これは、b のシグナルが a のシグナルよりも左にある（先に発生している）
    # ことを意味します。
    #
    # 逆に、estimated_delay が正の場合、
    # a に対して b を遅らせた場合に相互相関が最大となります。
    # これは、b のシグナルが a のシグナルよりも右にある（後に発生している）
    # ことを意味します。
    estimated_delay = corr.argmax() - (len(b) - 1)
    return [corr, estimated_delay]
