from operator import itemgetter

from matplotlib import pyplot as plt

from config import base_model_dir, experiment2_dir, base_plot_dir
from evaluate.evaluate_model import evaluate_model
from evaluate.load_model import load_model_from_checkpoint
from model.cnn_for_mnist import build_cnn_for_mnist
from model.patch_transformer import build_patch_transformer
from train_util import calculate_trainable_variables
from util import get_params


def evaluate_models():
    image_size = 28
    num_classes = 10
    params = get_params('model', 'experiment2')
    patch_size = params['patch_size']

    params = get_params('model', 'experiment2', 'cnn')
    model_dir = base_model_dir / experiment2_dir
    accuracies = {}
    number_of_params = {}
    for param_key in params.keys():
        dir_string = f'cnn-{param_key}'
        directory = model_dir / dir_string
        cnn = build_cnn_for_mnist(num_classes=num_classes, **params[param_key])
        cnn = load_model_from_checkpoint(cnn, directory)
        acc = evaluate_model(cnn)
        accuracies[dir_string] = acc
        params_count = calculate_trainable_variables(cnn)
        number_of_params[dir_string] = params_count

    params = get_params('model', 'experiment2', 'patch')
    for param_key in params.keys():
        dir_string = f'patch-{param_key}'
        directory = model_dir / dir_string
        patch_transformer = build_patch_transformer(
            num_classes=num_classes, image_size=image_size,
            patch_size=patch_size, **params[param_key])
        patch_transformer = load_model_from_checkpoint(patch_transformer,
                                                       directory)
        acc = evaluate_model(patch_transformer)
        accuracies[dir_string] = acc
        params_count = calculate_trainable_variables(patch_transformer)
        number_of_params[dir_string] = params_count
    return accuracies, number_of_params


def plot_accuracies(acc, n_params, filename):
    cnn_values = []
    patch_values = []
    for key, value in acc.items():
        num_params = n_params[key]
        if key.startswith('cnn'):
            cnn_values.append((num_params, value))
        else:
            patch_values.append((num_params, value))

    cnn_values.sort(key=itemgetter(0))
    patch_values.sort(key=itemgetter(0))

    print('experiment 2')
    print(cnn_values)
    print(patch_values)

    plt.cla()
    plt.ylabel('accuracy')
    plt.xlabel('number of parameters')  # TODO count real parameters
    cnn_x = [cnn_val[0] for cnn_val in cnn_values]
    cnn_y = [cnn_val[1] for cnn_val in cnn_values]
    patch_x = [patch_val[0] for patch_val in patch_values]
    patch_y = [patch_val[1] for patch_val in patch_values]

    plt.plot(cnn_x, cnn_y, label='CNN', marker='.')
    plt.plot(patch_x, patch_y, label='7x7 Vision Transformer', marker='.')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(filename)


def experiment2():
    accuracies, number_of_params = evaluate_models()
    plot_accuracies(accuracies, number_of_params,
                    base_plot_dir / 'cnn-patch-comparison')
