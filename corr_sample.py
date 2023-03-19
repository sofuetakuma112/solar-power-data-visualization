from matplotlib import pyplot as plt
import numpy as np

FONT_SIZE = 14

if __name__ == "__main__":
    target_sig = np.random.normal(size=1000) * 1.0
    delay = 800
    sig1 = np.random.normal(size=2000) * 0.2
    sig1[delay : delay + 1000] += target_sig
    sig2 = np.random.normal(size=2000) * 0.2
    sig2[:1000] += target_sig

    corr = np.correlate(sig1, sig2, "full")
    estimated_delay = corr.argmax() - (len(sig2) - 1)

    figsize_px = np.array([1280, 720])
    dpi = 100
    figsize_inch = figsize_px / dpi

    fig, axes = plt.subplots(3, 1, figsize=figsize_inch, dpi=dpi)

    axes[0].set_ylabel(
        "sig1",
        fontsize=FONT_SIZE,
    )
    axes[0].plot(sig1)
    axes[0].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    axes[1].set_ylabel(
        "sig2",
        fontsize=FONT_SIZE,
    )
    axes[1].plot(sig2, color="g")
    axes[1].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    axes[2].set_ylabel(
        "fit",
        fontsize=FONT_SIZE,
    )
    axes[2].plot(np.arange(len(sig1)), sig1)
    axes[2].plot(np.arange(len(sig2)) + estimated_delay, sig2)
    axes[2].set_xlim([0, len(sig1)])
    axes[2].tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    plt.show()
