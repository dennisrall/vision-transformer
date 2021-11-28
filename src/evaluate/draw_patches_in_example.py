import image_slicer
import tensorflow as tf
from matplotlib import pyplot as plt

from config import base_plot_dir
from load_dataset import load_normalized_mnist


def split_into_patches(filename):
    image_slicer.slice(filename, 16)


def draw_example_with_patches():
    directory = base_plot_dir / 'patches'
    directory.mkdir(exist_ok=True)
    filename = directory / 'example.png'
    save_example(filename)
    split_into_patches(filename)
    plot_patches(directory)
    plot_patches_linear(directory)


def save_example(filename):
    _, test_ds = load_normalized_mnist()
    data = next(iter(test_ds))
    image = data['image']
    image = tf.image.grayscale_to_rgb(image)
    image = image.numpy()
    plt.cla()
    plt.imsave(filename, image)


def plot_patches(directory):
    rows, cols = count_tiles(directory)
    plt.cla()
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            filename = directory / f'example_{row + 1:02d}_{col + 1:02d}.png'
            image = plt.imread(filename)
            axs[row, col].imshow(image)
            axs[row, col].axis('off')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(directory / 'example_tiles', bbox_inches='tight')


def plot_patches_linear(directory):
    rows, cols = count_tiles(directory)
    plt.cla()
    plt.margins(tight=True)
    fig, axs = plt.subplots(1, rows * cols)
    for row in range(rows):
        for col in range(cols):
            filename = directory / f'example_{row + 1:02d}_{col + 1:02d}.png'
            image = plt.imread(filename)
            axs[row * cols + col].imshow(image)
            axs[row * cols + col].axis('off')
    fig.tight_layout()
    plt.savefig(directory / 'example_tiles_linear', bbox_inches='tight')


def count_tiles(directory):
    tiles = []
    for file in directory.iterdir():
        if file.name in {'example.png', 'example_tiles.png',
                         'example_tiles_linear.png'}:
            continue
        row, col = file.name[-9:-7], file.name[-6:-4]
        tiles.append((int(row), int(col)))
    rows = max(t[0] for t in tiles)
    cols = max(t[1] for t in tiles)
    return rows, cols


if __name__ == '__main__':
    draw_example_with_patches()
