"""
[Title] mi_estimators.py
[Description] A file unifying various mi estimators.
              Codes are adapted and modified from existing papers's implementaions,
              added a lot more human readable comments.
[Todos]
    - [x] adapt ICLR '17 of simple binning: github.com/artemyk/ibsgd
    - [ ] write new estimator of binning by quantiles
    - [ ] adapt ICLR '19 of adaptive binning: github.com/ivansche/adaptive_estimators
    - [ ] adapt Schwartz-Ziv & Tishby's binning and kde: github.com/ravidziv/IDNNs
    - [ ] adapt standard MI estimation toolbox: github.com/gregversteeg/NPEET
    - [ ] adapt ICLR '20 sota variational approach: github.com/ermongroup/smile-mi-estimator
    - [ ] unify the function names and make a easy, general call with params
[Notes]
    - 🔥 denotes the lines needing potential future re-edits.
"""

from scipy.special import logsumexp
import numpy as np

# ############################################################################
# 0. Simplest Binning (mostly from ICLR '17)
# ############################################################################
# High-level: This approach uses the same bin size of each layer/dimension/epoch.
# Merits:
#   1. Fast to calculate, easy to implement.
# Drawbacks:
#   1. The same bin size across layers may hide the compression pattern for early layers.
#   2. The fixed binning range (e.g., [-1, 1]) cannot handdle non-saturating activations
#     like ReLU, which may hide the compression patterns as well. Need to well choose the
#     bin boundaries.
#   3. Noticeable  errors when working with continous Ys.
#   4. No guarantee on bias or variance.
#   5. May not maintain information processing properties.


def get_unique_probs(x):
    """
    Get the probability for each unique hidden outputs for a certain layer.
    The data fed in usually should be already binned.

    Inputs:
        x: (np.array) shape=(n_examples, n_dim), dtype=int
            This is the processed data from the output of a layer h.
            Each data entry is essentially the number of bins for the original value.
            For example, if h = [[12, 14], [15, 18]], bin_size = 6,
            then x = [[2, 2], [2, 3]]; all entries falls in the same bin except for the last one.

    Return:
        p_ts: (np.array) shape=(n_unique, )
              The probability (frequency) for each unique binned (scaled) hidden output in the sample
        unique_inverse: (np.array) the index used to reconstrcut the original array
                        check the np.unique() doctring for details
    """
    # The next line maks each row in the array as a string
    # No idea why do such complicated things while we can use boolean
    # References: stackoverflow.com/questions/16216078/test-for-membership-in-a-2d-numpy-array
    # The high level is that we want to check if any rows are of the same value
    # The shape of x_flatten is (n_samples,)
    # 🔥 Need to change this line to accommodate data with shape (n_examples,)
    x_flatten = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))

    # Here, x_unique is an array of unique entries (rows/instances) in x_flatten
    # The shape of x_unique is (n_unique_entry, ), typically longer if bin_size is smaller
    x_unique, unique_inverse, unique_counts = np.unique(x_flatten,
                                                        return_index=False,
                                                        return_inverse=True,
                                                        return_counts=True)

    # Here we calculate the probability for each unique entry
    # The shape of p_ts is the same as x_unique, i.e., (n_unique_entry, )
    p_ts = np.asarray(unique_counts / float(sum(unique_counts)))

    return p_ts, unique_inverse


def bin_calc_information(input_data, layer_data, num_of_bins, bin_min, bin_max):
    """
    A general function calculating mutual information.

    Inputs:
        input_data: (np.array) shape=(n_examples, n_data_dim)
        layer_data: (np.array) shape=(n_examples, n_hidden_dim)
        num_of_bins: (int) the number of bins
        bin_min: (float) the start value of bins, usually -1 if using tanh as activation
        bin_max: (float) the end value of bins, usually -1 if using tanh as activation
    """
    # Get the probability for the input data (or output data)
    # If x is continuous, n_unique = n_examples, no binning applied
    # meaning that p_xs = 1 / n_examples, corresponding to uniform distribution
    p_xs, unique_inverse_x = get_unique_probs(input_data)

    # Get the probability for the layer data
    # 🔥 This gets dangerous when working with non-saturating functions
    # 🔥 Using quantile function to do finegrained binning here
    bins = np.linspace(- 1, 1, num_of_bins, dtype='float32')

    # Get the indice for output after filering by bins
    ts_binned_inds = np.digitize(layer_data.reshape(- 1), bins) - 1
    ts_binned_inds[ts_binned_inds == - 1] = 0  # Avoid the error when ind = 0 but shifted to - 1 by above

    # Get the digitized data for the layer data
    digitized = bins[ts_binned_inds].reshape(len(layer_data), - 1)

    # Get the probability distribution for the layer data
    p_ts, _ = get_unique_probs(digitized)

    # Calculate H(T)
    h_layer = - np.sum(p_ts * np.log(p_ts))

    # Calculate H(T|X) = p(X=x_1) * H(T|X=x_1) + p(X=x_2) * H(T|X=x_2) + ...
    h_layer_given_input = 0.
    for xval in np.arange(len(p_xs)):
        # This is the extract rows of the layer data corresponding to the input data with same rows
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        h_layer_given_input += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return h_layer - h_layer_given_input


def bin_calc_information2(labelixs, layer_data, binsize):
    """
    Calculating I(X; T) and I(Y; T) when Y is *binary*.

    Inputs:
        labelixs (dict): data label correspondance, an instance is:
                 {0: array([False, True, False]),
                  1: array([True, False, True])}
        layer_data (np.array): the output of each layer;
                    the shape is (n_examples, n_dims)
        binsize (float): the pre-assigned bin size, which you can think of like a scale
                         notice that the bin size is the same for every dimension

    Returns:
        I(X; T): (float) This is just H(T) since I(X; T) = H(T) - H(T|X) = H(T)
        I(Y; T): (float) Calculated by I(Y; T) = H(Y) - H(Y|T)
    """
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        """
        Get the entropy of a layer output by binning and - sum(p * log(p)).

        Inputs:
            d (np.array): the same as layer_data with shape=(n_examples, n_dims)
                          usually the output of each layer.

        Return:
            The entropy for this layer's output.
        """
        # Next we convert the original output to binned output
        digitized = np.floor(d / binsize).astype('int')  # This step can be problematic
        p_ts, _ = get_unique_probs(digitized)
        return - np.sum(p_ts * np.log(p_ts))

    # Calculate I(X; T), i.e., H(T)
    h_layer = get_h(layer_data)

    # Calculate H(T|Y) = p(Y=0) * H(T|Y=0) + p(Y=1) * H(T|Y=1)
    h_layer_given_output = 0
    for label, ixs in labelixs.items():
        h_layer_given_output += ixs.mean() * get_h(layer_data[ixs, :]) # It uses the same binning size again.

    return h_layer, h_layer - h_layer_given_output


# ############################################################################
# 1. Entropy-based Adaptive Binning Estimator (ICLR '19)
# ############################################################################
# High-level: This approach adaptively chooses the bin boundaries such that each
#             bin contains the same number of unique observed activation levels.
# Merits:
#   1. Fast to calculate, easy to implement.
#   2. Handles non-saturaing activations (ReLU) well.
# Drawbacks:
#   1. Noticeable errors when working with continous Ys.
#   2. No guarantee on bias or variance.
#   3. May not maintain information processing properties.


# ############################################################################
# 2. Adaptive KDE Estimator (ICLR '19)
# ############################################################################
# High-level: Estimate density first, then calculate the information.
#             It adds isotropic Gaussian noise compared to traditional KDE.
#             Var(noise) is proportional to the max activity value in a given layer.
# Merits:
#   1. Handles non-saturaing activations (ReLU) well.
# Drawbacks:
#   1. The KDE process is a bit slow.
#   2. No guarantee on bias or variance.
#   3. May not maintain information processing properties.



# ############################################################################
# 3. Smoothed Lower Bound Estimator (ICLR '20)
# ############################################################################
# High-level: Estimating the lower bound directly for mutual information.
# Merits:
#   1. Guarantee on bias or variance, and handles its tradeoff.
#   2. Tight bound on the estimation.
#   3. Maintains some of the information processing properties.
# Drawbacks:
#   1. Unsure how it adapts to non-saturating activations.
#   2. A lower bound is always a "lower" bound
