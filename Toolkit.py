from pandas.api.types import is_datetime64_any_dtype as is_datetime
from os import path
import pandas as pd
import numpy as np
import datetime


def colour_palette():
    return ['#008AFE', '#6B54D9', '#AB15AC', '#EE0082', '#FE0053']


def colour_scheme(group):
    """
    Simple script that returns a hex colour and label based on a group string
    """

    # c_dict = {'discharged': ['#300a5c', 'Discharged (no ICU)'], 'ICU': ['#c83f4b', 'ICU'], 'died': ['#fcab10', 'Died on ward']}
    palette = colour_palette()
    c_dict = {'discharged': [palette[0], 'Discharged (no ICU)'], 'ICU': [palette[2], 'Discharged after ICU'], 'died': [palette[4], 'Deceased']}

    if group in c_dict:
        return c_dict[group][0], c_dict[group][1]
    else:
        print(f"Warning: unrecognised group string '{group}'. Default colour was returned (blue).")
        return 'blue', 'Unknown'


def data_before_ic(df, patient_df, time_col='time'):
    """
    Filters out lab- and other data that was gather for a patient after ICU admission
    :param df: Lab or other data. Must have 'pseudo_id' and time_col columns, where the latter is relative to 1st day of symptoms
    :param patient_df: pandas DataFrame or path string
    :return: filtered df
    """

    # If patient_df is a path to the file, load the file
    if isinstance(patient_df, str):
        patient_df = exclude("Stats/Patient_Statistics_LUMC.csv", "Stats/Patient_Statistics_LUMC.csv")
        patient_df.set_index('pseudo_id', inplace=True)
        patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
        patient_df['admitted_to_IC'] = pd.to_datetime(patient_df['admitted_to_IC'])

    if 'time_adm' in df.columns or 'start_time_adm' in df.columns:
        time_adm = 'time_adm' if 'time_adm' in df.columns else 'start_time_adm'

        # Add time column relative to ICU admission
        df['time_ICU'] = -1
        for pseudo_id in df['pseudo_id'].unique():
            if patient_df.at[pseudo_id, 'group_ICU'] == 1:
                time_of_ICU = patient_df.at[pseudo_id, 'admitted_to_IC_after_hours'] if not pd.isna(patient_df.at[pseudo_id, 'admitted_to_IC']) \
                    else float('inf')
                for index in df[df['pseudo_id'] == pseudo_id].index:
                    df.loc[index, 'time_ICU'] = df.loc[index, time_adm] - time_of_ICU

        df = df[df['time_ICU'] < 0]
        df = df.drop(columns=['time_ICU'])
    else:
        # If time column is absolute
        if is_datetime(df[time_col]):
            rel_time_col = time_col
        else:
            # Add time column that holds time points relative to admission and not day of symptoms
            df, rel_time_col = time_rel_adm(df, patient_df, time_col)
        for pseudo_id in df['pseudo_id'].unique():
            if not is_datetime(df[time_col]):
                time_to_ic_admission = patient_df.at[pseudo_id, 'admitted_to_IC_after_hours'] if not pd.isna(patient_df.at[pseudo_id, 'admitted_to_IC']) \
                    else float('inf')
            else:
                time_to_ic_admission = patient_df.at[pseudo_id, 'admitted_to_IC'] if not pd.isna(patient_df.at[pseudo_id, 'admitted_to_IC']) else \
                    datetime.datetime(2099, 1, 1)
            # Filter out rows
            df = df[((df['pseudo_id'] == pseudo_id) & (df[rel_time_col] < time_to_ic_admission))  # only rows for this patient before IC admission
                    | (df['pseudo_id'] != pseudo_id)]  # and all other rows for other patients left in this dataframe

        if not is_datetime(df[time_col]):
            df = df.drop(columns=[rel_time_col])

    return df


def exclude(df, patient_stats, admission_df="covid-studie-rachel-lumc-opname-delen-admission-parts.csv",
            admission_date=True, treatment_limit=True, discharge_dest=True, origin_other_hospital=True, dis_other_hospital=True, icu=True, seh_to_icu=True, only_seh=True,
            symptom_onset=True, single_department=True):
    """
    Reads Pandas dataframe and drops rows/columns based on exclusion criteria
    :param df: path to csv file (Pandas dataframe from where patients/features need to be excluded)
    :param patient_stats: path to csv key file (e.g. Patient_Statistics_LUMC.csv) that includes general data on all patients
    :param admission_df: path to csv file with raw admission data
    :param remaining params: turn on/off filters
    :return: Pandas dataframe
    """

    df = pd.read_csv(df)
    # Drop unwanted index column (if present)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Gather pseudo IDs of all patients that need to be excluded
    gen_df = pd.read_csv(patient_stats)
    adm_df = pd.read_csv(admission_df)
    adm_df['opname_delen_admission_parts_start_date'] = pd.to_datetime(adm_df['opname_delen_admission_parts_start_date'])
    adm_df['opname_delen_admission_parts_end_date'] = pd.to_datetime(adm_df['opname_delen_admission_parts_end_date'])
    # Add pseudo id counts
    adm_df['count'] = adm_df.apply(lambda row: adm_df['pseudo_id'].value_counts()[row['pseudo_id']], axis=1)
    # add time diff
    adm_df['time_diff'] = adm_df.apply(lambda row: (row['opname_delen_admission_parts_end_date'] - row['opname_delen_admission_parts_start_date']).total_seconds() / 3600, axis=1)

    ids = []
    # Exclude patients without admission date
    if admission_date:
        ids.extend(list(gen_df[pd.isna(gen_df['admitted_date'])]['pseudo_id']))
    # Exclude patients with treatment limitation(s)
    if treatment_limit:
        ids.extend(list(gen_df[(gen_df['no_IC'] == 1) | (gen_df['no_intubate'] == 1) | (gen_df['no_resuscitate'] == 1)]['pseudo_id']))
    # Exclude patients with an unknown discharge location
    if discharge_dest:
        ids.extend(list(gen_df[pd.isna(gen_df['discharge_dest'])]['pseudo_id']))
    # Exclude patients originating from another hospital
    if origin_other_hospital:
        ids.extend(list(gen_df[(~pd.isna(gen_df['admission_origin'])) & (gen_df['admission_origin'].str.contains('ziekenhuis', case=False))]['pseudo_id']))
    # Exclude patients that were discharged to another hospital / hospice
    if dis_other_hospital:
        ids.extend(list(gen_df[(~pd.isna(gen_df['discharge_dest'])) & ((gen_df['discharge_dest'].str.contains('ziekenhuis', case=False)) | (gen_df['discharge_dest'].str.contains('hospice', case=False)))]['pseudo_id']))
    # Exclude patients who were immediately admitted to the ICU
    if icu:
        ids.extend(list(gen_df[gen_df['admitted_loc'].isin(['IC', 'IC voor Volwassenen'])]['pseudo_id']))
    # Exclude those patients who were immediately admitted to the ICU from the SEH
    if seh_to_icu:
        ids.extend(list(gen_df[gen_df['SEH_to_ICU'] == 1]['pseudo_id']))
    # Exclude patients only admitted to the SEH (and who didn't die)
    if only_seh:
        ids.extend(list(gen_df[(gen_df['SEH_only'] == 1) & (gen_df['died'] == 0)]))
    # Exclude patients without a first symptoms day
    if symptom_onset and 'symptoms_date' in gen_df.columns:
        ids.extend(list(gen_df[pd.isna(gen_df['symptoms_date'])]['pseudo_id']))
    # Exclude patients only admitted to verloskunde or dagbehandeling and for less than 24 hrs
    if single_department:
        ids.extend(list(adm_df[(adm_df['count'] == 1) & ((adm_df['opname_delen_admission_parts_department'] == 'Verloskunde') | (adm_df['opname_delen_admission_parts_tags'] == 'Dagbehandeling')) &
                               (adm_df['time_diff'] < 24)]['pseudo_id']))
    # Filter out pseudo IDs from df
    df = df[~df['pseudo_id'].isin(ids)]

    return df


def get_baseline(data, patient_df, cutoff_times, time_col='time'):
    """
    Takes a Pandas dataframe and extracts baseline measurements for every patient based on certain time cutoff values
    :param data: Pandas Dataframe where two columns are 'pseudo_id' and time_col (order is irrelevant)
    :param patient_df: Pandas Dataframe (used for determining how long after admission a measurement was taken
    :param cutoff_times: list of cutoff times in hours after admission, 1 value for each variable
    :return: Pandas Dataframe with pseudo IDs in the index, and 1 column per feature. 1 row per patient
    """

    if len(cutoff_times) != len(data.columns) - 2:
        raise ValueError(f"Number of cutoff times does not match expected length (expected {len(data.columns) - 2}, got {len(cutoff_times)})")

    # Get unique patients in data df
    patients = data['pseudo_id'].unique()

    # Make output df
    o_df = pd.DataFrame(index=patients, columns=[i for i in data.columns if i not in ['pseudo_id', time_col]])

    # Add time column that holds time points relative to admission and not day of symptoms
    if time_col.endswith('_adm'):
        adm_time_col = time_col
    else:
        data, adm_time_col = time_rel_adm(data, patient_df)

    # Loop through features
    for i, col in enumerate([i for i in data.columns if i not in ['pseudo_id', time_col, adm_time_col]]):
        # Loop through all individual patients
        for pseudo_id in patients:

            # filter out relevant rows
            sub_data = data[(data['pseudo_id'] == pseudo_id)
                            & (data[adm_time_col] <= cutoff_times[i])
                            & (~pd.isna(data[col]))
                            ].reset_index(drop=True)

            if len(sub_data.index) > 0:
                o_df.loc[pseudo_id, col] = sub_data.loc[0, col]
            else:
                o_df.loc[pseudo_id, col] = np.nan
    return o_df


def granularity(df, col=0, interval_val=1, interval_unit='H', aggregation='mean', patient_denstity=True, print_progress=True):
    """
    Applies granularity over a Pandas dataframe.
    This function expects either datetime objects, or numeric relative times in the time column. In the case of the latter, interval_unit is not used
    :param df: Original Pandas dataframe
    :param col: Index of column in df that contains time values
    :param interval_val: Time interval value. Default = 1
    :param interval_unit: Time interval unit, comes from Pandas offset aliases:
                          https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    :param aggregation: Aggregation function. Decides how a series is condensed into a single value. Options: mean (default), median, min, max, sum
                        and density. If set to density, this function will return unique patient counts at each time interval
    :param patient_denstity: In case of aggregation='density', return the patient density (number of patients in a timeframe),
                             or return the number of observations for a patient in that timeframe
    :param print_progress: Print progression yes/no
    :return: Pandas dataframe o_df
    """

    pd.set_option("precision", 3)

    if print_progress:
        print(f"Applying granularity of {interval_val}{interval_unit}...")

    # Create an initial data table with entries from start till end time, with steps of size granularity.
    # Check time type (datetime or numeric) and make timestamps based on that
    if isinstance(df.iloc[0, col], datetime.datetime) and df.iloc[0, col] != df.iloc[-1, col]:
        if len(df.index) > 1:
            timestamps = pd.date_range(df.iloc[0, col], df.iloc[-1, col], freq=str(interval_val)+interval_unit)
        else:
            # WARNING: only works when interval_unit is set to 'H'
            timestamps = [df.iloc[0, col], df.iloc[0, col] + datetime.timedelta(hours=interval_val)]
    else:
        if len(df.index) > 1 and df.iloc[0, col] != df.iloc[-1, col]:
            # range() can't deal with floats. Convert to integers (*10), and then later divide into floats again (/10)
            timestamps = list(range(int(df.iloc[0, col] * 10), int(df.iloc[-1, col] * 10), interval_val * 10))
            timestamps = [i / 10 for i in timestamps]
        else:
            timestamps = [df.iloc[0, col], df.iloc[0, col] + interval_val]
    o_df = pd.DataFrame(columns=[c for c in df.columns if c != 'pseudo_id'])
    o_df['time'] = timestamps

    # Fill o_df
    for i, row in enumerate(o_df['time']):
        # Print progression
        if print_progress:
            if i % max(round(len(o_df.index) / 5), 1) == 0:
                print(f"Progress: row {i} / {len(o_df.index)} ({round(i / len(o_df.index) * 100, 1)}%)")

        if row == o_df['time'].iloc[-1]:
            if isinstance(df.iloc[0, col], datetime.datetime):
                time_offset = datetime.timedelta(weeks=999)
            else:
                time_offset = float('inf')
            upper_boundary = row + time_offset
        else:
            upper_boundary = o_df['time'].iloc[i + 1]

        # Give some leniency to the boundaries, to overcome floating point errors caused by rounding in Reformat_LUMC.py
        if not isinstance(df.iloc[0, col], datetime.datetime):
            upper_boundary += 0.05
            row -= 0.05

        # Select the relevant measurements from the df
        relevant_rows = df[(df[df.columns[col]] >= row) &
                           (df[df.columns[col]] < upper_boundary)]

        # Aggregate rows within timeframe
        if len(relevant_rows) > 0:
            for feature in [j for j in o_df if j not in ['time', 'pseudo_id']]:
                if aggregation == 'mean':
                    o_df.loc[i, feature] = relevant_rows[feature].mean()
                elif aggregation == 'median':
                    o_df.loc[i, feature] = relevant_rows[feature].median()
                elif aggregation == 'min':
                    o_df.loc[i, feature] = relevant_rows[feature].min()
                elif aggregation == 'max':
                    o_df.loc[i, feature] = relevant_rows[feature].max()
                elif aggregation == 'sum':
                    o_df.loc[i, feature] = relevant_rows[feature].sum()
                elif aggregation == 'density':
                    if patient_denstity:
                        o_df.loc[i, feature] = len(relevant_rows[~relevant_rows[feature].isna()]['pseudo_id'].unique())
                    else:
                        o_df.loc[i, feature] = len(relevant_rows[~relevant_rows[feature].isna()].index)
                else:
                    raise ValueError(f"Unknown aggregation: '{aggregation}'")
        else:
            for feature in [j for j in o_df if j not in ['time', 'pseudo_id']]:
                o_df.loc[i, feature] = 0

    return o_df


def save_file(obj, fpath, **kwargs):
    """
    Saves an object to a specified path
    :param obj: object to be saved
    :param fpath: full or relative path to save file to
    :param **kwargs: extra arguments used while saving the file
    """

    # Get file extension
    extension = path.splitext(fpath)[1]

    # Save file depending on extention
    if extension == '.csv':
        obj.to_csv(fpath, encoding='utf-8-sig', **kwargs)
        print("Saved file:", fpath)
    elif extension in ['.png', '.jpg']:
        obj.savefig(fpath, **kwargs)
        print("Saved figure:", fpath)


def time_rel_adm(df, patient_df, time_col='time'):
    """
    Takes a time column and makes it relative to day of admission instead
    Expects a 'pseudo_id' and 'time' column in 'df'
    The time column can be relative (to day of symptoms) or absolute
    :param df: input data with time relative to start of symptoms
    :param patient_df: patient_df
    :param time_col: optional, name of column containing time values
    :return: Pandas Dataframe df with added column
    """

    # Make dict with time to admission for every patient
    time_to_admission = {}
    for pseudo_id in df['pseudo_id'].unique():
        time_to_admission[pseudo_id] = (patient_df.at[pseudo_id, 'admitted_date'] - patient_df.at[pseudo_id, 'symptoms_date']).total_seconds() / 3600

    for row in df.index:
        pseudo_id = df.at[row, 'pseudo_id']
        df.loc[row, f'{time_col}_rel_adm'] = df.at[row, time_col] - time_to_admission[pseudo_id]

    return df, f'{time_col}_rel_adm'


