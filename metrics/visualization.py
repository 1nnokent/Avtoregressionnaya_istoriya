from metrics.generation import PIXEL_MAX, PIXEL_MIN

GRID_SIDE = 8
GRID_CELL_INCHES = 1.0
GRID_DPI = 120


def save_grid(images, path, side=GRID_SIDE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = side * GRID_CELL_INCHES
    fig, axes = plt.subplots(side, side, figsize=(size, size))
    for ax, image in zip(axes.flat, images):
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="gray", vmin=PIXEL_MIN, vmax=PIXEL_MAX)
        else:
            ax.imshow(image.transpose(1, 2, 0).clip(PIXEL_MIN, PIXEL_MAX))
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=GRID_DPI)
    plt.close(fig)