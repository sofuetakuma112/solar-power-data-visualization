def init_rcParams(rcParams, font_size=12, figsize=[6.4, 4.8], dpi=100):
    rcParams["figure.figsize"] = figsize
    rcParams["figure.dpi"] = dpi

    rcParams["axes.labelsize"] = font_size
    rcParams["xtick.labelsize"] = font_size
    rcParams["ytick.labelsize"] = font_size
    rcParams["axes.titlesize"] = font_size
    rcParams["legend.fontsize"] = font_size

    return rcParams


def figsize_px_to_inch(figsize_px, dpi=100):
    return figsize_px / dpi
