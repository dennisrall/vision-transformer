from config import base_plot_dir
from evaluate.draw_patches_in_example import draw_example_with_patches
from evaluate.attention_visualization import visualize_attention
from evaluate.experiment1 import experiment1
from evaluate.experiment2 import experiment2
from evaluate.experiment3 import experiment3
from evaluate.experiment4 import experiment4
from util import get_params

if __name__ == '__main__':
    base_plot_dir.mkdir(exist_ok=True)
    experiment1()
    experiment2()
    experiment3()
    experiment4()
    draw_example_with_patches()
    visualize_attention(**get_params('evaluate', 'attention_visualization'))
