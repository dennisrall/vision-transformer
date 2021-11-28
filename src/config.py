from pathlib import Path

# files and directories
params_file = 'params.yaml'
base_output_dir = Path('out/')
base_train_log_dir = base_output_dir / 'log'
base_val_log_dir = base_output_dir / 'log'
base_model_dir = base_output_dir / 'model'
base_plot_dir = base_output_dir / 'plots'
naive_dir = Path('naive')
patch_dir = Path('patch')
cnn_dir = Path('cnn')
experiment1_dir = Path('experiment1')
experiment2_dir = Path('experiment2')
experiment3_dir = Path('experiment3')
experiment4_dir = Path('experiment4')

# experiment 1
transformer_blocks = [5, 10, 15, 20, 25]
patch_sizes = [2, 7]
