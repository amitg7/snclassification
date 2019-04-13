from astropy.io import fits
import os
import numpy as np
import pandas as pd

# The data will be normalized to this range of log_M
LOG_M_RANGE = np.linspace(8, 12, 200)


# Load all .cat files from the folder, and returns DataFrame contains all of them
def load_cat_table(data_folder_path, dir_names):
    cat_table = {}
    for dir_name in dir_names:
        dir_path = os.path.join(data_folder_path, dir_name)
        cat_file_name = filter(lambda fn: fn.endswith('.cat'), os.listdir(dir_path))[0]
        cat_table[dir_name] = pd.read_table(
            os.path.join(dir_path, cat_file_name),
            usecols=['OBJECT', 'REDSHIFT', 'TYPE', 'FILE'],
            delim_whitespace=True,
            index_col=0)

    full_cat_table = pd.DataFrame()
    for dir_name, df in cat_table.iteritems():
        full_cat_table = full_cat_table.append(df.assign(DIR_NAME=dir_name))

    return full_cat_table


# Filter the cat table by the transient types and redshift
def filter_cat_table(cat_table, transient_types, redshift_upper_bound=1.):
    cat_table = cat_table.dropna()
    cat_table = cat_table[cat_table.REDSHIFT < redshift_upper_bound]
    cat_table = cat_table[cat_table.TYPE.isin(transient_types)]

    return cat_table


# Transform the given full cat table to a dict
def create_cat_table_dict(full_cat_table):
    cat_table_dict = {}
    transient_types = full_cat_table.TYPE.unique()

    for transient_type in transient_types:
        cat_table_dict[transient_type] = full_cat_table[full_cat_table.TYPE == transient_type]

    return cat_table_dict


# Load all objects from the path names in the cat table, save them in dest_dir, and return as a dict.
# If they are already exist, load them and return as a dict.
def load_all_objects_by_transient_type(cat_table_dict, data_dir_path, dest_dir):
    transient_types_folder_path = dest_dir

    transient_type_dict = {}

    if not os.path.isdir(transient_types_folder_path):
        os.makedirs(transient_types_folder_path)

    for transient_type in cat_table_dict.iterkeys():
        print('Loading "{}"... ({} objects)'.format(transient_type, cat_table_dict[transient_type].shape[0]))

        csv_file_path = os.path.join(transient_types_folder_path, 'df_{}.csv'.format(transient_type))

        if os.path.exists(csv_file_path):

            # Load from csv file
            transient_type_df = pd.DataFrame.from_csv(csv_file_path)

        else:

            # Load from fits file (raw)
            transient_type_df = pd.DataFrame()
            for object_id, row in cat_table_dict[transient_type].iterrows():
                file_path = os.path.join(data_dir_path, row.DIR_NAME, row.FILE)
                with fits.open(file_path) as object_data:
                    object_df = pd.DataFrame(object_data['POSTERIOR PDF'].data)
                    object_df.insert(0, 'object_id', value=object_id)
                    transient_type_df = transient_type_df.append(object_df)

            # Save to csv file
            csv_file_path = os.path.join(transient_types_folder_path, 'df_{}.csv'.format(transient_type))
            transient_type_df.to_csv(csv_file_path)

        transient_type_dict[transient_type] = transient_type_df

    print('Done.')

    return transient_type_dict
