"""
[Title] plotter.py
[Description] The function will be directly called in run.py to generate plots.
"""
from pathlib import Path
import torch
import joblib
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np


# ##########################################################
# Helper Function to smooth lines
# ##########################################################
def smooth(scalars,
           weight: float=0.6):
    """
    This helps the generated line looks more smoothed.
    Reference: stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot

    Inputs:
        scalars: (list or array) the list of values you'd like to smoothed out
        weight: (float) the weight for the smooth level;
                        the larger, the more smooth the output line looks like

    Returns:
        smoothed: (list) the smoothed outputs
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        # Calculate smoothed value
        smoothed_val = last * weight + (1 - weight) * point
        # Save it
        smoothed.append(smoothed_val)
        # Anchor the last smoothed value
        last = smoothed_val

    return smoothed


# ##########################################################
# Plot SNR
# ##########################################################
def plot_gradients(final_path: str='./final_path',
                   grad_plot_path: str='./final_path/gradients.pdf',
                   grad_plot_caption: str='Input your caption.',
                   save_fisher: int=1,
                   smooth_weight: float=0.6,
                   log_scale: int=1,
                   draw_weight_norm: int=0):
    """
    Generating a one-row figure, containing each layer's gradient information,
    as well as the loss and accuracy plots.

    We use log scale for x-axis.

    Inputs:
        final_path: (str/path) the path saving the pickled file
        fig_path: (str) the path to save the figure; usually defined in run.py.
        caption: (str) the necessary information like network structure, lr, etc.
                 This is usually just the folder name.
        smooth_weight: (float) the parameter to smooth the lines, 0 if no smooth.

    Returns:
        None. You will have your image saved in fig_path.
    """

    # Get the statistics from the model
    grad_dict = torch.load(Path(final_path) / 'grad_dict.pkl')
    results_dict = joblib.load(Path(final_path) / 'results.pkl')
    loss_train_list = [i.cpu() for i in results_dict['loss_train_list']]
    loss_test_list = [i.cpu() for i in results_dict['loss_test_list']]
    acc_train_list = [i.cpu() for i in results_dict['acc_train_list']]
    acc_test_list = [i.cpu() for i in results_dict['acc_test_list']]

    # Use a dumb way to get number of layers
    num_layers = len(grad_dict['epoch 0']['grad_mean'])
    print('Num layers:', num_layers)

    # Draw figures
    fig, axes = plt.subplots(1, num_layers + 3,
                             figsize=(4.5 * (num_layers + 1), 5.5))
    for i in range(num_layers):
        # Extract mean, std, and snr
        grad_mean = torch.tensor([value['grad_mean'][i]
                             for value in grad_dict.values()])
        grad_std = torch.tensor([value['grad_std'][i]
                            for value in grad_dict.values()])
        weight_norm = torch.tensor([value['weight_norm'][i]
                            for value in grad_dict.values()])
        snr = grad_mean / grad_std

        # Plot
        axes[i].plot(smooth(snr, smooth_weight), color='red',
                     label=f'grad SNR')
        axes[i].plot(smooth(grad_mean, smooth_weight), color='black',
                     label=f'grad mean')
        axes[i].plot(smooth(grad_std, smooth_weight), color='gray',
                     label=f'grad std', linestyle='-.')
        if draw_weight_norm:
            axes[i].plot(smooth(weight_norm, smooth_weight), color='blue',
                         label=f'weight norm', linestyle=':')

        # Set the title
        axes[i].set_title(f'Layer {i + 1}', fontsize=20)

    # Draw the fisher plot
    if save_fisher:
        fisher_dict = torch.load(Path(final_path)  / 'fisher_dict.pkl')
        axes[- 3].plot(list(fisher_dict.keys()), list(fisher_dict.values()),
                       color='green')
        axes[- 3].set_ylabel(r'Tr$(\mathbf{F})$', fontsize=20)
        axes[- 3].set_title(r'Trace of Fisher Information')

    # Draw the loss plot
    axes[- 2].plot(loss_train_list, color='hotpink', label=f'train loss')
    axes[- 2].plot(loss_test_list, color='blueviolet',
                   linestyle='-.', label=f'test loss')
    axes[- 2].set_title('Loss', fontsize=20)

    # Draw the final loss point
    best_tr_loss = np.array(loss_train_list).min()
    best_te_loss = np.array(loss_test_list).min()
    axes[- 2].plot(np.array(loss_train_list).argmin() - 1, best_tr_loss, 'o',
                   color='hotpink', label=f'best train: {best_tr_loss:.3f}')
    axes[- 2].plot(np.array(loss_test_list).argmin() - 1, best_te_loss, 'o',
                   color='blueviolet', label=f'best test: {best_te_loss:.3f}')
    # The - 1 here is to debug, in case it does not show.

    # Draw the accuracy plot
    axes[- 1].plot(acc_train_list, color='deeppink', label=f'train accuracy')
    axes[- 1].plot(acc_test_list, color='midnightblue',
                   linestyle='-.', label=f'test accuracy')
    axes[- 1].set_title('Accuracy', fontsize=20)
    axes[- 1].set_ylim(0.5, 1.0)

    # Draw the final accuracy point
    best_tr_acc = np.array(acc_train_list).max()
    best_te_acc = np.array(acc_test_list).max()
    axes[- 1].plot(np.array(acc_train_list).argmax(), best_tr_acc, 'o',
                   color='deeppink',
                   label=f'best train: {best_tr_acc * 100:.2f}%')
    axes[- 1].plot(np.array(acc_test_list).argmax(), best_te_acc, 'o',
                   color='midnightblue',
                   label=f'best test: {best_te_acc * 100:.2f}%')

    # Set the legend
    axes[0].legend(loc='center left', fontsize=13, framealpha=0.3)
    axes[- 2].legend(loc='upper right', fontsize=13, framealpha=0.3)
    axes[- 1].legend(loc='lower right', fontsize=13, framealpha=0.3)


    # Set the common tick label size, log scale, and xlabel
    for i in range(num_layers + 3):
        axes[i].xaxis.set_tick_params(labelsize=20)
        axes[i].yaxis.set_tick_params(labelsize=20)
        axes[i].set_xlabel('no. of epochs', fontsize=20)
        if log_scale:
            axes[i].set_xscale('log', nonpositive='clip')
            if i < num_layers:
                axes[i].set_yscale('log', nonpositive='clip')

    # Caption
    plt.figtext(0.5, 0.01, grad_plot_caption, wrap=True,
                horizontalalignment='center', fontsize=12)
    fig.subplots_adjust(bottom=0.18)
    sea.despine()
    plt.tight_layout()
    plt.savefig(grad_plot_path)

    return None


# ##########################################################
# Plot General Accuracy and Loss
# ##########################################################
def plot_performance(final_path: str='./final_path',
                     performance_plot_path: str='./final_path/performance.pdf',
                     grad_plot_caption: str='Input your caption.'):
    """
    The plot for accuracy and loss of the model during training.
    """

    # Get the statistics from the model
    results_dict = joblib.load(Path(final_path) / 'results.pkl')
    loss_train_list = [i.cpu() for i in results_dict['loss_train_list']]
    loss_test_list = [i.cpu() for i in results_dict['loss_test_list']]
    acc_train_list = [i.cpu() for i in results_dict['acc_train_list']]
    acc_test_list = [i.cpu() for i in results_dict['acc_test_list']]

    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # --------------- Plot for loss --------------- #
    # Draw the loss plot
    axes[0].plot(loss_train_list, color='hotpink', label=f'train loss')
    axes[0].plot(loss_test_list, color='blueviolet',
                   linestyle='-.', label=f'test loss')
    axes[0].set_title('Loss', fontsize=20)

    # Draw the final loss point
    best_tr_loss = np.array(loss_train_list).min()
    best_te_loss = np.array(loss_test_list).min()
    axes[0].plot(np.array(loss_train_list).argmin() - 1, best_tr_loss, 'o',
                   color='hotpink', label=f'best train: {best_tr_loss:.3f}')
    axes[0].plot(np.array(loss_test_list).argmin() - 1, best_te_loss, 'o',
                   color='blueviolet', label=f'best test: {best_te_loss:.3f}')
    # The - 1 here is to debug, in case it does not show.

    # --------------- Plot for Accuracy --------------- #
    # Draw the accuracy plot
    axes[1].plot(acc_train_list, color='deeppink', label=f'train accuracy')
    axes[1].plot(acc_test_list, color='midnightblue',
                   linestyle='-.', label=f'test accuracy')
    axes[1].set_title('Accuracy', fontsize=20)
    axes[1].set_ylim(0.5, 1.0)

    # Draw the final accuracy point
    best_tr_acc = np.array(acc_train_list).max()
    best_te_acc = np.array(acc_test_list).max()
    axes[1].plot(np.array(acc_train_list).argmax(), best_tr_acc, 'o',
                   color='deeppink',
                   label=f'best train: {best_tr_acc * 100:.2f}%')
    axes[1].plot(np.array(acc_test_list).argmax(), best_te_acc, 'o',
                   color='midnightblue',
                   label=f'best test: {best_te_acc * 100:.2f}%')

    # Set the legend
    axes[0].legend(loc='upper left', fontsize=15, framealpha=0.3)
    axes[1].legend(loc='lower left', fontsize=15, framealpha=0.3)

    # Set the label and ticks
    for i in range(2):
        axes[i].set_xlabel('no. of epochs', fontsize=20)
        axes[i].xaxis.set_tick_params(labelsize=15)
        axes[i].yaxis.set_tick_params(labelsize=15)

    # Caption
    # plt.figtext(0.5, 0.01, grad_plot_caption, wrap=True,
    #             horizontalalignment='center', fontsize=12)
    # fig.subplots_adjust(bottom=0.18)
    sea.despine()

    # Save figure
    plt.tight_layout()
    plt.savefig(performance_plot_path)

    return None


# ##########################################################
# Plot Accuracy and Loss for Noisy data
# ##########################################################
def plot_performance_noisy(final_path: str='./final_path',
                           performance_noisy_plot_path: str='.',
                           grad_plot_caption: str='.'):
    """
    The plot for accuracy and loss of the model during training.
    """

    # Get the statistics from the model
    results_dict = joblib.load(Path(final_path) / 'results.pkl')
    loss_clean_list = [i.cpu() for i in results_dict['loss_clean_list']]
    loss_noisy_list = [i.cpu() for i in results_dict['loss_noisy_list']]
    acc_clean_list = [i.cpu() for i in results_dict['acc_clean_list']]
    acc_generalized_list = [i.cpu() for i in results_dict['acc_generalized_list']]
    acc_memorized_list = [i.cpu() for i in results_dict['acc_memorized_list']]

    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --------------- Plot for loss --------------- #
    # Draw the loss plot
    axes[0].plot(loss_clean_list, color='hotpink', label=f'loss on clean data')
    axes[0].plot(loss_noisy_list, color='blueviolet',
                   linestyle='-.', label=f'loss on noisy data')
    axes[0].set_title('Loss', fontsize=20)

    # --------------- Plot for Accuracy on Clean Data --------------- #
    # Draw the accuracy plot
    axes[1].plot(acc_clean_list, color='green', label=f'correct')
    axes[1].set_title('Acc on Clean Data', fontsize=20)
    axes[1].set_ylim(0.0, 1.0)

    # --------------- Plot for Accuracy on Noisy data --------------- #
    # Draw the accuracy plot
    axes[2].plot(acc_generalized_list, color='green', label=f'correct')
    axes[2].plot(acc_memorized_list, color='red',
                   linestyle='-.', label=f'memorized')
    axes[2].set_title('Acc on Noisy Data', fontsize=20)
    axes[2].set_ylim(0.0, 1.0)


    # Set the legend
    axes[0].legend(loc='upper left', fontsize=15, framealpha=0.3)
    axes[1].legend(loc='lower left', fontsize=15, framealpha=0.3)
    axes[2].legend(loc='lower left', fontsize=15, framealpha=0.3)

    # Set the label and ticks
    for i in range(3):
        axes[i].set_xlabel('no. of epochs', fontsize=20)
        axes[i].xaxis.set_tick_params(labelsize=15)
        axes[i].yaxis.set_tick_params(labelsize=15)

    # Caption
    # plt.figtext(0.5, 0.01, grad_plot_caption, wrap=True,
    #             horizontalalignment='center', fontsize=12)
    # fig.subplots_adjust(bottom=0.18)
    sea.despine()

    # Save figure
    plt.tight_layout()
    plt.savefig(performance_noisy_plot_path)

    return None


# ##########################################################
# Plot Fisher Information
# ##########################################################
def plot_fisher(final_path: str='./final_path',
                fisher_dict_path: str='./final_path/fisher_dict.pkl',
                fisher_plot_path: str='./final_path/fisher.pdf'):
    """
    The plot for trace of fisher information.
    """

    # Load data
    fisher_dict = torch.load(fisher_dict_path)

    # Handle the case when there is a string key
    if 'best' in fisher_dict.keys():
        fisher_dict.pop('best')

    # Draw figure
    fig = plt.figure(figsize=(8, 4))
    plt.plot(list(fisher_dict.keys()),
             list(fisher_dict.values()),
             color='black')
    plt.xlabel('no. of epochs', fontsize=20)
    plt.ylabel(r'Tr$(\mathbf{F})$', fontsize=20)
    sea.despine()

    # Save figure
    plt.tight_layout()
    plt.savefig(fisher_plot_path)

    return None


# ##########################################################
# Plot Weight Norms
# ##########################################################
def plot_weights(final_path: str='./final_path',
                 weight_plot_path: str='./final_path/weights.pdf',
                 weight_plot_caption: str='.',
                 smooth_weight: float=0.0):
    """
    Generating a one-row figure, containing each layer's weight information.

    We use log scale for x-axis.

    Inputs:
        model: (Model) the class we defined in ../optim/model.py.
        fig_path: (str) the path to save the figure; usually defined in run.py.
        caption: (str) the necessary information like network structure, lr, etc.
                 This is usually just the folder name.
        smooth_weight: (float) the parameter to smooth the lines, 0 if no smooth.

    Returns:
        None. You will have your image saved in fig_path.
    """

    # Get the statistics from the model
    grad_dict = torch.load(Path(final_path) / 'grad_dict.pkl')
    results_dict = joblib.load(Path(final_path) / 'results.pkl')
    loss_train_list = [i.cpu() for i in results_dict['loss_train_list']]
    loss_test_list = [i.cpu() for i in results_dict['loss_test_list']]
    acc_train_list = [i.cpu() for i in results_dict['acc_train_list']]
    acc_test_list = [i.cpu() for i in results_dict['acc_test_list']]

    # Use a dumb way to get number of layers
    num_layers = len(grad_dict['epoch 0']['grad_mean'])
    print('Num layers:', num_layers)

    # Draw figures
    fig, axes = plt.subplots(1, num_layers + 3,
                             figsize=(4.5 * (num_layers + 1), 5.5))
    for i in range(num_layers):
        # Extract mean, std, and snr
        weight_norm = torch.tensor([value['weight_norm'][i]
                            for value in grad_dict.values()])

        # Plot
        axes[i].plot(smooth(weight_norm, smooth_weight), color='black',
                     label=f'weight norm', linestyle=':')

        # Set the title
        axes[i].set_title(f'Layer {i + 1}', fontsize=20)

        # Draw the final loss point
        initial_weight_norm = weight_norm[0]
        final_weight_norm = weight_norm[- 1]
        axes[i].plot(0, initial_weight_norm, 'o',
                     color='green', label=f'initial weight: {initial_weight_norm:.3f}')
        axes[i].plot(len(weight_norm) - 1, final_weight_norm, 'o',
                     color='blue', label=f'final weight: {final_weight_norm:.3f}')

        # Set the legend
        axes[i].legend(loc='upper right', fontsize=13, framealpha=0.3)

    # Draw the loss plot
    axes[- 2].plot(loss_train_list, color='hotpink', label=f'train loss')
    axes[- 2].plot(loss_test_list, color='blueviolet',
                   linestyle='-.', label=f'test loss')
    axes[- 2].set_title('Loss', fontsize=20)

    # Draw the final loss point
    # final_tr_acc = loss_train_list[- 1]
    # final_te_acc = loss_test_list[- 1]
    best_tr_loss = np.array(loss_train_list).min()
    best_te_loss = np.array(loss_test_list).min()
    axes[- 2].plot(np.array(loss_train_list).argmin(), best_tr_loss, 'o',
                   color='hotpink', label=f'best train: {best_tr_loss:.3f}')
    axes[- 2].plot(np.array(loss_test_list).argmin(), best_te_loss, 'o',
                   color='blueviolet', label=f'best test: {best_te_loss:.3f}')

    # Draw the accuracy plot
    axes[- 1].plot(acc_train_list, color='deeppink', label=f'train accuracy')
    axes[- 1].plot(acc_test_list, color='midnightblue',
                   linestyle='-.', label=f'test accuracy')
    axes[- 1].set_title('Accuracy', fontsize=20)
    axes[- 1].set_ylim(0.5, 1.0)

    # Draw the final accuracy point
    # final_tr_acc = acc_train_list[- 1]
    # final_te_acc = acc_test_list[- 1]
    best_tr_acc = np.array(acc_train_list).max()
    best_te_acc = np.array(acc_test_list).max()
    axes[- 1].plot(np.array(acc_train_list).argmax(), best_tr_acc, 'o',
                   color='deeppink',
                   label=f'best train: {best_tr_acc * 100:.2f}%')
    axes[- 1].plot(np.array(acc_test_list).argmax(), best_te_acc, 'o',
                   color='midnightblue',
                   label=f'best test: {best_te_acc * 100:.2f}%')

    # Set the legend
    axes[- 2].legend(loc='upper right', fontsize=15, framealpha=0.3)
    axes[- 1].legend(loc='lower right', fontsize=15, framealpha=0.3)


    # Set the common tick label size, log scale, and xlabel
    for i in range(num_layers + 3):
        axes[i].xaxis.set_tick_params(labelsize=20)
        axes[i].yaxis.set_tick_params(labelsize=20)
        axes[i].set_xlabel('no. of epochs', fontsize=20)

    # Caption
    plt.figtext(0.5, 0.01, weight_plot_caption, wrap=True,
                horizontalalignment='center', fontsize=12)
    fig.subplots_adjust(bottom=0.18)
    sea.despine()
    plt.tight_layout()
    plt.savefig(weight_plot_path)

    return None
