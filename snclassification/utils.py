import numpy as np


def calculate_weighted_mean_mass(df):
    return np.log10((df['probability'] * 10**df['mass']).sum() / df['probability'].sum())


# Return the DataFrame grouped by the object_id, with the weighted mean mass
def calculate_object_mean_mass(transient_type_df):
    return transient_type_df.groupby('object_id').apply(calculate_weighted_mean_mass)


# def calculate_weighted_std_mass(df):
#     weighted_mean_mass = calculate_weighted_mean_mass(df)
#     n = df.shape[0]
#     if n <= 1:
#         return 0
#     return np.sqrt(n * (df['probability'] * (df['mass'] - weighted_mean_mass)**2).sum() /
#                    ((n-1)*df['probability'].sum()))


# def calculate_mean_std_mass(df):
#     n = df.shape[0]
#     if n == 0:
#         return 0
#     return np.sqrt((df['mass_std']**2).mean() / np.sqrt(n))


# # Return the DataFrame grouped by the object_id, with the weighted std mass
# def calculate_object_std_mass(transient_type_df):
#     return transient_type_df.groupby('object_id').apply(calculate_weighted_std_mass)


# Returns the bin centers from the bin edges
def get_bin_centers(bin_edges):
    width = np.diff(bin_edges)[0]
    return (bin_edges[:-1] + bin_edges[:-1] + width) / 2.


# A normalization function
def normalize(values, delta):
    norm_factor = np.sum(values) * delta
    return values / norm_factor
