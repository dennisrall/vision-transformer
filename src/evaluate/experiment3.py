from matplotlib import pyplot as plt

from config import base_model_dir, experiment3_dir, base_plot_dir
from evaluate.evaluate_model import evaluate_model
from evaluate.load_model import clean_up_params, load_model_from_checkpoint
from model.hybrid_transformer import build_hybrid_transformer, build_pure_cnn
from util import get_params


def evaluate_models():
    params = get_params('model', 'experiment3')
    params = clean_up_params(params, num_classes=10)

    model_dir = base_model_dir / experiment3_dir
    accuracies = {}
    for x in model_dir.iterdir():
        dir_string = x.name
        if dir_string == 'pure_cnn':
            continue
        _, _, num_cnn_layers = dir_string.partition('-')
        params['num_cnn_layers'] = int(num_cnn_layers)
        directory = model_dir / dir_string
        model = build_hybrid_transformer(**params)
        model = load_model_from_checkpoint(model, directory)
        acc = evaluate_model(model, use_rgb_mnist=True)
        accuracies[dir_string] = acc
    model = build_pure_cnn(params['ff_dim'], params['num_classes'])
    model = load_model_from_checkpoint(model, model_dir / 'pure_cnn')
    acc = evaluate_model(model, use_rgb_mnist=True)
    accuracies['pure_cnn'] = acc
    accuracies.pop('vgg16-19', None)
    return accuracies


def plot_accuracies(accuracies, filename):
    pure_cnn_val = accuracies['pure_cnn']
    transformer_values = [
        (key, value)
        for key, value in accuracies.items()
        if key != 'pure_cnn'

    ]

    def key_func(item):
        _, _, num_cnn_layers = item[0].partition('-')
        return int(num_cnn_layers)

    transformer_values.sort(key=key_func)

    plt.cla()
    plt.ylabel('accuracy')
    plt.xlabel('number of CNN layers')
    num_cnn_layers = [key_func(t) for t in transformer_values]
    acc_values = [t[1] for t in transformer_values]
    print('experiment 3')
    *num_cnn_layers, _last_layer = num_cnn_layers
    *acc_values, _last_layer = acc_values
    print(num_cnn_layers)
    print(acc_values)

    plt.plot(num_cnn_layers, acc_values, label='hybrid transformer',
             marker='.')
    plt.axhline(pure_cnn_val, label='pre-trained CNN', color='red')
    plt.xticks(list(range(1, 18, 2)))
    plt.legend()
    plt.savefig(filename)


def experiment3():
    accuracies = evaluate_models()
    plot_accuracies(accuracies, base_plot_dir / 'hybrid_transformer')


if __name__ == '__main__':
    experiment3()
