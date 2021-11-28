import matplotlib.pyplot as plt

from config import base_plot_dir, base_model_dir, experiment4_dir
from evaluate.evaluate_model import evaluate_model
from evaluate.load_model import load_model_from_checkpoint
from model.transfer_learning_transformer import \
    build_transfer_learning_transformer
from util import get_params


def evaluate_models():
    params = get_params('model', 'experiment4')
    vit_types = params.pop('vit_types')

    model_dir = base_model_dir / experiment4_dir

    image_size = 32
    num_classes = 10
    accuracies = {}

    for vit_type in vit_types:
        directory = model_dir / vit_type
        model = build_transfer_learning_transformer(image_size, vit_type,
                                                    num_classes)
        model = load_model_from_checkpoint(model, directory)
        acc = evaluate_model(model, use_rgb_mnist=True)
        accuracies[vit_type] = acc
    return accuracies


def experiment4():
    accuracies = evaluate_models()
    print('experiment 4')
    print(accuracies)
    plot_accuracies(accuracies,
                    base_plot_dir / 'transfer_learning_transformer')


def plot_accuracies(accuracies, filename):
    plt.cla()
    plt.ylabel('accuracy')
    plt.ylim(0.75, 1)
    plt.xlabel('different pre-trained vision transformer')
    plt.bar(accuracies.keys(), accuracies.values())
    plt.savefig(filename)


if __name__ == '__main__':
    experiment4()
