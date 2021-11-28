from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt

from config import transformer_blocks, patch_sizes, base_plot_dir, \
    base_model_dir, experiment1_dir
from evaluate.flops import get_flops
from evaluate.load_model import clean_up_params, \
    build_and_call_naive_transformer, build_and_call_patch_transformer, \
    load_model_from_checkpoint
from model.naive_transformer import build_naive_transformer
from model.patch_transformer import build_patch_transformer
from train_util import calculate_trainable_variables
from util import get_params
from evaluate.evaluate_model import evaluate_model


def count_flops():
    image_shape = (28, 28, 1)
    params = clean_up_params(get_params('model', 'experiment1'),
                             num_classes=10,
                             image_shape=image_shape)

    def count_naive_flops(num_transformer_blocks):
        params['num_transformer_blocks'] = num_transformer_blocks
        model = build_and_call_naive_transformer(**params)
        return get_flops(model)

    def count_patch_flops(num_transformer_blocks):
        params['num_transformer_blocks'] = num_transformer_blocks
        model = build_and_call_patch_transformer(**params)
        return get_flops(model)

    flops = {'naive': [
        count_naive_flops(num_transformer_blocks)
        for num_transformer_blocks in transformer_blocks
    ]}

    del params['image_shape']
    params['image_size'] = 28

    for patch_size in patch_sizes:
        name = f'{patch_size}x{patch_size}-patches'
        params['patch_size'] = patch_size
        flops[name] = [
            count_patch_flops(num_transformer_blocks)
            for num_transformer_blocks in transformer_blocks
        ]
    return flops


def get_label_and_marker(name):
    if '2' in name:
        return '2x2 Vision Transformer', 'd'
    elif '7' in name:
        return '7x7 Vision Transformer', 'x'
    return 'naive Transformer', '.'


def plot_flops(flops, filename, color_dict=None):
    if color_dict is None:
        color_dict = {}
    plt.cla()
    plt.ylabel('FLOPs')
    plt.xlabel('number of transformer layers')
    for name, values in flops.items():
        label, marker = get_label_and_marker(name)
        plt.plot(transformer_blocks, values, label=label,
                 marker=marker, color=color_dict.get(name, None))
    plt.xticks(transformer_blocks)
    plt.legend()
    plt.savefig(filename)


def fix_colors(iterable):
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return dict(zip(iterable, cycle(default_colors)))


def make_flop_plots():
    flops = count_flops()
    fixed_colors = fix_colors(flops.keys())
    plot_flops(flops, base_plot_dir / 'flops', fixed_colors)
    del flops['naive']
    plot_flops(flops, base_plot_dir / 'flops_patch', fixed_colors)
    del flops['2x2-patches']
    plot_flops(flops, base_plot_dir / 'flops_patch7x7', fixed_colors)


def evaluate_models(epochs, batch_size, embed_dim, num_heads, ff_dim):
    image_size = 28
    num_classes = 10

    model_dir = base_model_dir / experiment1_dir

    accuracies = defaultdict(list)
    num_params = defaultdict(list)

    for num_transformer_blocks in transformer_blocks:
        key = 'naive'
        dir_string = f'naive-{num_transformer_blocks}layers'
        directory = model_dir / dir_string

        image_shape = (image_size, image_size, 1)
        naive = build_naive_transformer(image_shape, embed_dim, num_heads,
                                        ff_dim, num_classes,
                                        num_transformer_blocks)
        naive = load_model_from_checkpoint(naive, directory)
        acc = evaluate_model(naive)
        params = calculate_trainable_variables(naive)
        accuracies[key] += acc,
        num_params[key] += params,

        for patch_size in patch_sizes:
            key = f'patch{patch_size}'
            dir_string = f'patch{patch_size}-{num_transformer_blocks}layers'
            directory = model_dir / dir_string

            patch_transformer = build_patch_transformer(
                patch_size, image_size, embed_dim, num_heads, ff_dim,
                num_classes, num_transformer_blocks)
            patch_transformer = load_model_from_checkpoint(patch_transformer,
                                                           directory)
            acc = evaluate_model(patch_transformer)
            params = calculate_trainable_variables(patch_transformer)
            accuracies[key] += acc,
            num_params[key] += params,
    return accuracies, num_params


def plot_accuracies(accuracies, filename):
    plt.cla()
    plt.ylabel('accuracy')
    plt.xlabel('number of transformer layers')
    for name, values in accuracies.items():
        _, patch, n = name.partition('patch')
        if patch == 'patch':
            name = f'{n}x{n}-patches'
        label, marker = get_label_and_marker(name)
        plt.plot(transformer_blocks, values, label=label, marker=marker)
    plt.xticks(transformer_blocks)
    plt.legend()
    plt.savefig(filename)


def plot_num_params(num_params, filename):
    plt.cla()
    plt.ylabel('number of parameters')
    plt.xlabel('number of transformer layers')
    for name, values in num_params.items():
        _, patch, n = name.partition('patch')
        if patch == 'patch':
            name = f'{n}x{n}-patches'
        label, marker = get_label_and_marker(name)
        plt.plot(transformer_blocks, values, label=label, marker=marker)
    plt.xticks(transformer_blocks)
    plt.legend()
    plt.savefig(filename)


def make_accuracy_plot(**params):
    accuracies, num_params = evaluate_models(**params)
    print('experiment 1')
    print(accuracies)
    print(num_params)
    plot_accuracies(accuracies, base_plot_dir / 'accuracies')
    plot_num_params(num_params, base_plot_dir / 'num_params')


def experiment1():
    make_flop_plots()
    params = get_params('model', 'experiment1')
    make_accuracy_plot(**params)


if __name__ == '__main__':
    experiment1()
