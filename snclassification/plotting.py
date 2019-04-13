import matplotlib.pyplot as plt
import seaborn as sns
from .utils import calculate_object_mean_mass


def plot_redshift_histograms(transient_type_dict):
    fig = plt.figure(figsize=(16, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, (transient_type, df) in enumerate(transient_type_dict.iteritems()):
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_title('{} ({})'.format(transient_type, df.shape[0]))
        sns.distplot(df.redshift, kde=False, bins=10, ax=ax, norm_hist=False)
    plt.show()


def plot_weighted_mass_histograms(transient_type_dict, bins, save_to_file=None):
    fig = plt.figure(figsize=(16, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (transient_type, df) in enumerate(transient_type_dict.iteritems()):
        ax = fig.add_subplot(2, 5, i + 1)

        # Weighted mean
        weighted_mean_mass = calculate_object_mean_mass(df)
        ax.hist(weighted_mean_mass, bins=bins, density=True, color='gray')

        ax.set_title('{} ({})'.format(transient_type, weighted_mean_mass.shape[0]))
        ax.set_xlim(bins.min(), bins.max())

    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight', dpi=600, transparent=True)

    plt.text(-0.3, -0.25, '$\log (M/M_{\odot})$', ha='center', fontsize=20)
    plt.show()


def plot_weighted_mass_histograms_and_mass_function_SFR(transient_type_dict, bins, mass_function_SFR, log_M_range):
    fig = plt.figure(figsize=(16, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (transient_type, df) in enumerate(transient_type_dict.iteritems()):
        ax = fig.add_subplot(2, 5, i + 1)

        # Weighted mean
        weighted_mean_mass = calculate_object_mean_mass(df)
        sns.distplot(weighted_mean_mass,
                     kde=False, bins=bins, ax=ax, norm_hist=True, color='yellow')

        # The mass function (from the papers)
        plt.plot(log_M_range, mass_function_SFR(log_M_range), c='r')

        ax.set_title('{} ({})'.format(transient_type, weighted_mean_mass.shape[0]))

    plt.figlegend(['Mass function weighted by SFR', 'Mass histogram (data)'], loc=6)
    plt.show()
