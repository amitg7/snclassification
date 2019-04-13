import sympy as sp
import numpy as np
import pandas as pd
from .data import LOG_M_RANGE
from .fitting import BIN_WIDTH


# Best-fit Double-Schechter Parameters
# Table 2 from Tomczak et al. (2014)
# http://iopscience.iop.org/article/10.1088/0004-637X/783/2/85/pdf
def schechter_parameters_from_tomczak(table_file_path):
    parameters_raw = pd.read_table(table_file_path, skiprows=14, nrows=3)

    def extract_values(col_name):
        return pd.to_numeric(parameters_raw[col_name].str.extract('([-+\.\d]+)', expand=False))

    parameters = pd.DataFrame(parameters_raw.Redshift.str.split(' < z < ').tolist(),
                              columns=['Redshift_from', 'Redshift_to'], dtype=float)
    parameters['log(M)'] = extract_values('Log(M*)')
    parameters['log(phi_1)'] = extract_values('Log()')
    parameters['alpha_1'] = extract_values('alpha_1')
    parameters['log(phi_2)'] = extract_values('Log().1')
    parameters['alpha_2'] = extract_values('alpha_2')
    return parameters


# Best-fit Schechter Function Parameters
# Table 1 from Muzzin (2013)
# http://iopscience.iop.org/article/10.1088/0004-637X/777/1/18/pdf
def schechter_parameters_from_muzzin(table_file_path):
    # # We chose the same parameters for every redshift
    # muzzin_parameters_dict = {
    #     'log(M)': 10.81,
    #     'log(phi_1)': np.log10(11.35e-4),
    #     'alpha_1': -1.34,
    #     'log(phi_2)': -np.inf,
    #     'alpha_2': 0
    # }
    # muzzin_parameters = parameters.copy()
    # muzzin_parameters.update(pd.DataFrame(muzzin_parameters_dict, index=[0, 1, 2]))
    # muzzin_parameters

    parameters_raw = pd.read_table(table_file_path, skiprows=3)

    row_numbers = [7, 14]
    col_names = parameters_raw.columns[[4, 5, 6]]

    def extract_values(col_name):
        return pd.to_numeric(parameters_raw[col_name].str.extract('([-+\.\d]+)', expand=False)).loc[row_numbers]

    parameters = pd.DataFrame(parameters_raw.Redshift.str.split(' <or= z < ').tolist(),
                              columns=['Redshift_from', 'Redshift_to'], dtype=float).loc[row_numbers]

    parameters['log(M)'] = extract_values(col_names[0])
    parameters['log(phi_1)'] = np.log10(extract_values(col_names[1]) * 10**(-4.))
    parameters['alpha_1'] = extract_values(col_names[2])
    parameters['log(phi_2)'] = -np.inf
    parameters['alpha_2'] = 0
    return parameters


def evaluate_double_schecter_functions(parameters):
    M, phi_1, phi_2, M_star, alpha_1, alpha_2 = sp.symbols(
        ['M', r'\phi_1', r'\phi_2', 'M^{*}', r'\alpha_1', r'\alpha_2'])

    phi = sp.log(10) * sp.exp(-10 ** (M - M_star)) * 10 ** (M - M_star) * \
          (phi_1 * 10 ** (alpha_1 * (M - M_star)) + phi_2 * 10 ** (alpha_2 * (M - M_star)))

    phi_dict = {}

    for _, row in parameters.iterrows():
        phi_reduced = phi.subs({
            M_star: row['log(M)'],
            phi_1: 10**row['log(phi_1)'],
            phi_2: 10**row['log(phi_2)'],
            alpha_1: row.alpha_1,
            alpha_2: row.alpha_2
        })
        redshift_from = round(float(row.Redshift_from), 2)
        redshift_to = round(float(row.Redshift_to), 2)
        phi_dict[(redshift_from, redshift_to)] = sp.lambdify(M, phi_reduced)

    return phi_dict


def evaluate_single_schecter_functions(parameters):
    M, phi_1, phi_2, M_star, alpha_1, alpha_2 = sp.symbols(
        ['M', r'\phi_1', r'\phi_2', 'M^{*}', r'\alpha_1', r'\alpha_2'])

    phi = sp.log(10) * sp.exp(-10 ** (M - M_star)) * phi_1 * 10 ** ((M - M_star) * (1 + alpha_1))

    phi_dict = {}

    for _, row in parameters.iterrows():
        phi_reduced = phi.subs({
            M_star: row['log(M)'],
            phi_1: 10**row['log(phi_1)'],
            phi_2: 10**row['log(phi_2)'],
            alpha_1: row.alpha_1,
            alpha_2: row.alpha_2
        })
        redshift_from = round(float(row.Redshift_from), 2)
        redshift_to = round(float(row.Redshift_to), 2)
        phi_dict[(redshift_from, redshift_to)] = sp.lambdify(M, phi_reduced)

    return phi_dict


# Returns log(SFR) as a function of logM and z (galaxy main-sequence)
# Whitaker (2014) http://iopscience.iop.org/article/10.1088/2041-8205/754/2/L29/pdf
def generate_log_SFR_func():
    log_M_star, z = sp.symbols([r'\log{M_*}', 'z'])
    log_SFR = (0.7 - 0.13*z) * (log_M_star - 10.5) + 0.38 + 1.14 * z - 0.19 * z**2
    log_SFR_func = sp.lambdify([log_M_star, z], log_SFR)
    return log_SFR_func


# Returns the mass function weighted by SFR, for the given redshift z
def generate_mass_function_SFR(phi_dict, z, normed=True):

    # Get the corresponding mass function, for the given redshift z
    redshift_from, redshift_to = list(filter(lambda z_bounds: z_bounds[0] <= z < z_bounds[1], phi_dict.keys()))[0]
    phi_func = phi_dict[(redshift_from, redshift_to)]

    # Get SFR function
    log_SFR_func = generate_log_SFR_func()

    # The mass function weighed by SFR, for the given z
    def mass_function_SFR(log_M):
        return phi_func(log_M) * 10**log_SFR_func(log_M, z)

    if normed:
        dM = (LOG_M_RANGE[1] - LOG_M_RANGE[0])
        normalization_factor = dM / BIN_WIDTH
        return lambda log_M: mass_function_SFR(log_M) / normalization_factor

    return mass_function_SFR


# Returns the mass function weighted by SFR and the efficiency function, for the given z
def generate_mass_function_SFR_efficiency(phi_dict, rho_func, z):

    # Get the mass function weighted by SFR
    mass_function_SFR = generate_mass_function_SFR(phi_dict, z)

    def mass_function_SFR_efficiency(log_M, log_M0, beta, C):
        efficiency_factor = rho_func(10 ** log_M, 10 ** log_M0, beta, C)
        return mass_function_SFR(log_M) * efficiency_factor

    return mass_function_SFR_efficiency
