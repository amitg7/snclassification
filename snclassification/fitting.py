import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .utils import get_bin_centers, calculate_object_mean_mass
from .data import LOG_M_RANGE

BIN_WIDTH = 0.5
BINS = np.arange(LOG_M_RANGE.min(), LOG_M_RANGE.max() + BIN_WIDTH, BIN_WIDTH)


def calculate_bin_coordinates(data, bins=None):
    bins = BINS if bins is None else bins
    hist, bin_edges = np.histogram(data, normed=True, bins=bins)

    bin_heights = hist / np.sum(hist)
    bin_centers = get_bin_centers(bin_edges)
    # bin_width = bins[1] - bins[0]

    return bin_centers, bin_heights


def generate_random_bin_edges():
    offset = np.random.uniform(0, BIN_WIDTH)
    return np.arange(BINS.min() + offset, BINS.max() - BIN_WIDTH + offset, BIN_WIDTH)


# Returns the DataFrame grouped by the object_id with sampled masses (=number_of_samples)
def create_group_by_object_sampled(df, number_of_samples):
    def create_sample(g):
        return g.mass.sample(n=number_of_samples, weights=g.probability, replace=True)

    return df.groupby('object_id').apply(create_sample).groupby(by='object_id')


class ParamProperties:
    def __init__(self, params, bounds, latex):
        self.params = params
        self.bounds = bounds
        self.latex = latex

    def get_bounds(self, params):
        bounds_dict = {p: b or (-np.inf, np.inf) for p, b in zip(self.params, self.bounds)}
        return [bounds_dict[p] for p in params]

    def get_latex(self, params):
        latex_dict = {p: l or p for p, l in zip(self.params, self.latex)}
        return [latex_dict[p] for p in params]

    def reduced_to(self, params):
        return ParamProperties(params, self.get_bounds(params), self.get_latex(params))


class FittingValues(dict):
    def __init__(self, popt, cost, params, success):
        dict.__init__(self, {param: value for value, param in zip(popt, params)})
        self.params = params
        self.cost = cost
        self.success = success

    def __getattr__(self, item):
        return self[item]

    def _calculate_chi_square(self):
        perr = np.vstack(np.sqrt(np.diag(self.cost)))
        cs = perr.T.dot(np.linalg.inv(self.cost).dot(perr))
        return np.asscalar(cs)


class FittingValuesList(dict):
    def __init__(self, instances):
        dict.__init__(self)
        self.params = instances[0].params
        self.costs = [instance.cost for instance in instances]
        self.success_arr = [instance.success for instance in instances]
        self.success = np.sum(self.success_arr)

        for instance in instances:
            self.append(instance)

        for param, values in self.iteritems():
            self[param] = np.array(values)

    def append(self, instance):
        for param, value in instance.iteritems():
            if param not in self:
                self[param] = []
            self[param].append(value)

    def __getattr__(self, item):
        return self[item]

    def get_success_bitmap(self, beta_threshold=0):
        return np.bitwise_and(np.array(self.success_arr, dtype=bool), self['beta'] > beta_threshold)

    def get_values(self, param, success_only=False):
        if success_only:
            return self[param][self.get_success_bitmap()]
        return self[param]

    def unpack_values(self, success_only=False):
        return [self.get_values(param, success_only) for param in self.params]

    def unpack_means(self, success_only=False):
        return [np.mean(self.get_values(param, success_only)) for param in self.params]

    def unpack_stds(self, success_only=False):
        return [np.std(self.get_values(param, success_only)) for param in self.params]

    def summary(self, params_in_latex=None, success_only=False):
        means = self.unpack_means(success_only)
        str_arr = []
        for i in range(len(self.params)):
            param_str = '${}$'.format(params_in_latex[i]) if params_in_latex else self.params[i]
            mean_value = means[i] if not np.isnan(means[i]) else '---'
            str_arr.append('{}={:<10.3}'.format(param_str, mean_value))
        return ''.join(str_arr)


class Fitting:
    def __init__(self, transient_type_dict, fitting_function, fitting_param_properties,
                 number_of_samples=None, random_bin_centers=False):
        self.transient_type_dict = transient_type_dict

        self.fitting_function = fitting_function
        self.fitting_params = fitting_param_properties.params
        self.fitting_params_bounds = fitting_param_properties.bounds
        self.fitting_params_latex = fitting_param_properties.latex

        self.number_of_samples = number_of_samples
        self.bootstrapped = bool(number_of_samples)

        self.random_bin_centers = random_bin_centers

        self.results = self.fit()

        self.bins = BINS

    # Returns the least_squares results in FittingValuesList
    def estimate_parameters_ls(self, data, bins=None):
        bin_centers, bin_heights = calculate_bin_coordinates(data, bins)

        bounds_null = zip(*(self.fitting_params_bounds[2:]))
        bounds = zip(*self.fitting_params_bounds)

        def residuals_null(params):
            values = self.fitting_function(bin_centers, np.inf, 1., *params)
            return bin_heights - values

        def residuals(params):
            values = self.fitting_function(bin_centers, *params)
            return bin_heights - values

        ls_res_null = least_squares(residuals_null, [lb for lb in bounds_null[0]], bounds=bounds_null)
        ls_res = least_squares(residuals, [lb for lb in bounds[0]], bounds=bounds)

        # Check whether the fit is better than the fit without an efficiency function
        success = int(ls_res.cost < ls_res_null.cost)

        return FittingValues(ls_res.x, ls_res.cost, self.fitting_params, success)

    def fit_one_transient_type(self, df):
        fitting_values_list = []

        # No bootstrapping
        if not self.bootstrapped:
            weighted_mean_mass = calculate_object_mean_mass(df)
            fitting_values = self.estimate_parameters_ls(weighted_mean_mass)
            fitting_values_list.append(fitting_values)

        # Bootstrapping
        else:

            # Sample a mass for each each object
            object_samples_groups = create_group_by_object_sampled(df, self.number_of_samples)

            # Iterate over samples
            for sample_num in range(self.number_of_samples):

                # Take on sample
                object_masses_sample = object_samples_groups.apply(lambda g: g.iat[sample_num])

                if self.random_bin_centers:
                    bins = generate_random_bin_edges()
                else:
                    bins = BINS

                fitting_values = self.estimate_parameters_ls(object_masses_sample, bins=bins)
                fitting_values_list.append(fitting_values)

        return FittingValuesList(fitting_values_list)

    def fit(self):
        results = {}
        for i, (transient_type, df) in enumerate(self.transient_type_dict.iteritems()):
            results[transient_type] = self.fit_one_transient_type(df)
        return results

    # region Results

    def print_results(self, success_only=False):
        for i, (transient_type, fitting_values) in enumerate(self.results.iteritems()):
            print('{:<10}{}'.format(transient_type, fitting_values.summary(success_only=success_only)))

    def plot_M0_and_beta_distribution(self, transient_type, save_to_file=None, success_only=False):
        fitting_values_list = self.results[transient_type]
        M0_vals, beta_vals = fitting_values_list.unpack_values(success_only)[:2]
        M0_latex, beta_latex = self.fitting_params_latex[:2]

        if np.alen(M0_vals) > 1:
            fit_results_fig = sns.jointplot(M0_vals, beta_vals, kind='scatter', size=4, space=0, s=5)
            fit_results_fig.set_axis_labels(r'$%s$' % M0_latex, r'$%s$' % beta_latex, fontsize=16)
            fit_results_fig.fig.suptitle(transient_type, x=0.1, fontsize=22)

            if save_to_file:
                plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)

    def plot_M0_and_beta_distributions(self, save_to_dir=None, success_only=False):
        for transient_type in self.transient_type_dict.iterkeys():
            if save_to_dir:
                if not os.path.isdir(save_to_dir):
                    os.makedirs(save_to_dir)
                save_to_file = os.path.join(save_to_dir, '%s.svg' % transient_type)
            else:
                save_to_file = None
            self.plot_M0_and_beta_distribution(transient_type,
                                               save_to_file=save_to_file, success_only=success_only)

    def plot_results_in_figure(self, plt, color=None, success_only=False):
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(self.results)))

        x_label_latex, y_label_latex = self.fitting_params_latex[:2]

        for i, (transient_type, fitting_values_list) in enumerate(self.results.iteritems()):
            curr_color = color if color is not None else colors[i]

            x_mean, y_mean = fitting_values_list.unpack_means(success_only)[:2]
            x_std, y_std = fitting_values_list.unpack_stds(success_only)[:2]

            if self.bootstrapped:
                plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                             linewidth=1, capsize=3, fmt='o', c=curr_color, label=transient_type)
            else:
                plt.scatter(x_mean, y_mean, linewidth=1, c=curr_color, label=transient_type)

            plt.annotate(xy=(x_mean, y_mean), s=transient_type,
                         ha='right', fontsize=16, textcoords='offset points', xytext=(-2, 5), color=curr_color)

        plt.xlabel(r'$%s$' % x_label_latex, fontsize=16)
        plt.ylabel(r'$%s$' % y_label_latex, fontsize=16)

        if color is None:
            plt.legend(fontsize=12)

        plt.grid(True)

    def plot_results(self, color=None, save_to_file=None, success_only=False):
        plt.figure(figsize=(9, 9))
        self.plot_results_in_figure(plt, color, success_only)
        if save_to_file:
            plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)
        plt.show()

    def plot_fitting_function_in_ax(self, ax, transient_type, success_only=False):

        # Fitting parameters
        fitting_values_list = self.results[transient_type]

        # Histogram
        df = self.transient_type_dict[transient_type]
        weighted_mean_mass = calculate_object_mean_mass(df)
        bin_centers, bin_heights = calculate_bin_coordinates(weighted_mean_mass)

        # Plot bin heights
        ax.scatter(bin_centers, bin_heights, c='r', label='Histogram bar heights')

        # Plot fitting
        log_M_range = LOG_M_RANGE
        ax.plot(log_M_range, self.fitting_function(log_M_range, *fitting_values_list.unpack_means(success_only)),
                c='w', linestyle='dashed', label='Fitting')
        ax.scatter(bin_centers, self.fitting_function(bin_centers, *fitting_values_list.unpack_means(success_only)),
                   c='w', s=20)

        ax.set_title(r'{:<10} $\quad$ {}'.format(transient_type,
                                                 fitting_values_list.summary(self.fitting_params_latex, success_only)),
                     fontsize=15)
        ax.set_xlabel(r'$\log (M/M_{\odot})$', fontsize=12)
        ax.legend()

    def plot_fitting_functions(self, save_to_file=None, success_only=False):
        fig = plt.figure(figsize=(15, 18))
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        for i, (transient_type, df) in enumerate(self.results.iteritems()):
            ax = fig.add_subplot(5, 2, i + 1)
            self.plot_fitting_function_in_ax(ax, transient_type, success_only)

        if save_to_file:
            plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)

        plt.show()

    @classmethod
    def plot_comparison(cls, fitting_instances, descriptions, cmap=None, success_only=False):
        plt.figure(figsize=(14, 14))

        if cmap:
            colors = [plt.get_cmap(cmap)(i) for i in range(len(fitting_instances))]
        else:
            colors = ['g', 'b', 'r', 'k', 'purple']

        legend_elements = []
        for i, fitting_instance in enumerate(fitting_instances):
            fitting_instance.plot_results_in_figure(plt, color=colors[i], success_only=success_only)
            legend_elements.append(
                plt.matplotlib.patches.Patch(color=colors[i], label=descriptions[i]))

        plt.legend(handles=legend_elements, fontsize=12)
        plt.show()

    @classmethod
    def plot_M0_histograms(cls, labels, fitting_instances, save_to_file=None, success_only=False):
        fig, axes = plt.subplots(10, 1, figsize=(10, 7), sharex=True)

        transient_types = sorted(fitting_instances[0].results.keys())
        number_of_samples = fitting_instances[0].number_of_samples
        bins = fitting_instances[0].bins

        for ax, transient_type in zip(axes, transient_types):
            captions = []

            for i, fitting_instance in enumerate(fitting_instances):
                M0_vals = fitting_instance.results[transient_type].unpack_values(success_only)[0]

                ax.hist(M0_vals, bins=100, range=(bins.min() - 3, bins.max() + 3),
                        alpha=0.6, color=plt.get_cmap('Set1')(len(fitting_instances) - i - 1))

                if success_only:
                    percentage = np.alen(M0_vals) / float(number_of_samples)
                    captions.append(r'{} ({:.1%})'.format(transient_type, percentage).replace('%', r'\%'))
                else:
                    captions.append(transient_type)

            ax.text(-0.02, 0.5, unicode(captions[-1]), fontsize=15, ha='right', va='center', transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xlim([bins.min() - 0.5, bins.max() + 0.5])

        axes[-1].set_xlabel(r'$\log (M_0/M_{\odot})$', fontsize=18)
        axes[-1].tick_params(axis='both', which='major', labelsize=16)
        fig.subplots_adjust(hspace=0)
        fig.legend(labels, ncol=2, loc=9, borderaxespad=1, fontsize=10)
        if save_to_file:
            plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)

        # fig.suptitle('Number of samples: %s' % number_of_samples)

    @classmethod
    def plot_stripplot(cls, labels, fitting_instances, save_to_file=None, success_only=False):
        use_tex = plt.rcParams['text.usetex']
        plt.rcParams['text.usetex'] = False

        transient_types = sorted(fitting_instances[0].results.keys())

        transient_type_mass_data_frames = []

        for fitting_instance, label in zip(fitting_instances, labels):
            for transient_type in transient_types:
                log_M0_arr = fitting_instance.results[transient_type].unpack_values(success_only=success_only)[0]
                df = pd.DataFrame({'transient_type': transient_type, 'log_M0': log_M0_arr, 'method': label})
                transient_type_mass_data_frames.append(df)

        full_df = pd.concat(transient_type_mass_data_frames)
        full_df = full_df.sort_values('method')
        full_df = full_df.sort_values('transient_type')

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sns.stripplot(x='log_M0', y='transient_type', hue='method',
                      palette='deep', data=full_df, dodge=True, jitter=False,
                      alpha=.08, zorder=1, size=7)

        # # Show the conditional means
        # sns.pointplot(x='log_M0', y='transient_type', hue='method',
        #               palette='pastel', data=full_df, dodge=.5, join=False,
        #               markers='d', scale=1.2, ci=None, color='#222222')

        ax.set_xlabel(r'$\log (M_0/M_{\odot})$', fontsize=16)
        ax.set_ylabel('Transient type', fontsize=16, position=(2, 0.5))

        # ax.set(xlabel=r'$\log (M_0/M_{\odot})$', ylabel='Transient type')
        ax.legend(fontsize=10, loc=9, bbox_to_anchor=(0.5, 1.2), ncol=2)
        ax.set_xlim([BINS.min() - 0.5, BINS.max() + 0.5])
        ax.grid(axis='x', color='w')

        if save_to_file:
            plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)

        plt.rcParams['text.usetex'] = use_tex

    # endregion
