import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from Toolkit import save_file


def load_data(c, m, l, a, o, q10, q11, vi, vms, mews):
    complaints_data = pd.read_csv(c, delimiter=',', quotechar='"')
    meds_data = pd.read_csv(m, delimiter=',', quotechar='"')
    lab_data = pd.read_csv(l, delimiter=',', quotechar='"')
    admission_data = pd.read_csv(a, delimiter=',', quotechar='"')
    overview_data = pd.read_csv(o, delimiter=',', quotechar='"')
    q10_data = pd.read_csv(q10, delimiter=',', quotechar='"')
    q11_data = pd.read_csv(q11, delimiter=',', quotechar='"')
    vitals_data = pd.read_csv(vi, delimiter=',', quotechar='"')
    vms_data = pd.read_csv(vms, delimiter=',', quotechar='"')
    mews_data = pd.read_csv(mews, delimiter=',', quotechar='"')

    # Add MEWS data to vitals data
    patient_df = pd.read_csv("Stats/Patient_Statistics_LUMC.csv")
    patient_df.set_index('pseudo_id', inplace=True)
    mews_data = mews_data[(mews_data['mews_form_entries_question'] == 'Totaalscore MEWS') & (mews_data['pseudo_id'].isin(patient_df.index))]
    mews_data.rename(columns={'mews_form_entries_question': 'vitale_metingen_observation_description',
                              'mews_form_entries_value_number': 'vitale_metingen_observation_test_value_numeric',
                              'mews_form_entries_start_date': 'vitale_metingen_observation_start_date'}, inplace=True)

    # Filter out MEWS scores from previous hospitalisations
    mews_data['vitale_metingen_observation_start_date'] = pd.to_datetime(mews_data['vitale_metingen_observation_start_date'])
    for pseudo_id in mews_data['pseudo_id'].unique():
        admitted_date = patient_df.at[pseudo_id, 'admitted_date']
        mews_data = mews_data[((mews_data['pseudo_id'] == pseudo_id) & (mews_data['vitale_metingen_observation_start_date'] >= admitted_date))
                              | (mews_data['pseudo_id'] != pseudo_id)]

    vitals_data = pd.concat([vitals_data, mews_data])


    return complaints_data, meds_data, lab_data, admission_data, overview_data, q10_data, q11_data, vitals_data, vms_data


def abs_to_rel_date(base_date, date):
    """
    Converts absolute datetime object to relative time
    :param base_date: time 0
    :param date: absolute date + time (string) to be converted
    :return: float, hours after admission
    """

    time_diff = round((pd.to_datetime(date) - pd.to_datetime(base_date)).total_seconds() / 3600, 1)
    return time_diff


def fill_time_gap(o_df, time_start, time_stop, cols=None, vals=None):
    """
    :param o_df: Pandas DataFrame that needs to be edited
    :param time_start: lower boundary of time gap
    :param time_stop: upper boundary of time gap
    :param cols: list of column names from df that need to be edited
    :param vals: list of values (relative to columns in cols) that needs to be added
    :return: edited DataFrame

    cols and vals need to be of the same length
    """
    if cols is not None and vals is not None:
        if len(cols) != len(vals):
            raise ValueError(f"Length of variable 'cols' ({len(cols)}) does not match with that of 'vals' ({len(vals)})")
    else:
        cols = []
        vals = []

    cols.insert(1, 'time')
    vals.insert(1, time_start)
    o_df = o_df.append(dict(zip(cols, vals)), ignore_index=True)
    time_start = pd.to_datetime(time_start)
    time_stop = pd.to_datetime(time_stop)
    time_diff = int((time_stop - time_start).total_seconds() / 3600)
    for hr in range(0, time_diff, 4):
        vals[1] = time_start + timedelta(hours=hr)
        o_df = o_df.append(dict(zip(cols, vals)), ignore_index=True)
    return o_df


def reformat_complaints(df, patient_data, symp_time=True):
    print("\nProcessing complaints data...")
    # Sort df
    df.sort_values(['pseudo_id', 'klachten_form_entries_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    cols = ['pseudo_id', 'time_symp', 'time_adm', 'type', 'value'] if symp_time else ['pseudo_id', 'time_adm', 'type', 'value']
    o_df = pd.DataFrame(columns=cols)

    # Loop through df
    for row in range(len(df.index)):
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")
        pseudo_id = df['pseudo_id'].iloc[row]

        if symp_time:
            time_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], df['klachten_form_entries_start_date'].iloc[row])
        time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], df['klachten_form_entries_start_date'].iloc[row])

        if symp_time:
            data = {'pseudo_id': pseudo_id, 'time_symp': time_symp, 'time_adm': time_adm, 'type': df['klachten_form_entries_question'].iloc[row],
                    'value': df['klachten_form_entries_value_text'].iloc[row]}
        else:
            data = {'pseudo_id': pseudo_id, 'time_adm': time_adm, 'type': df['klachten_form_entries_question'].iloc[row],
                    'value': df['klachten_form_entries_value_text'].iloc[row]}
        o_df = o_df.append(data, ignore_index=True)

    return o_df


def reformat_meds(df, patient_data, symp_time=True):
    print("\nProcessing medication data...")
    # Sort df
    df.sort_values(['pseudo_id', 'klinische_medicatie_medication_request_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    cols = ['pseudo_id', 'start_time_symp', 'end_time_symp', 'start_time_adm', 'end_time_adm', 'brandname', 'form', 'qty', 'unit', 'description'] if symp_time else \
           ['pseudo_id', 'start_time_adm', 'end_time_adm', 'brandname', 'form', 'qty', 'unit', 'description']
    o_df = pd.DataFrame(columns=cols)

    # Loop through df
    for row in range(len(df.index)):
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")
        pseudo_id = df['pseudo_id'].iloc[row]
        if symp_time:
            start_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], df['klinische_medicatie_medication_request_start_date'].iloc[row])
            end_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], df['klinische_medicatie_medication_request_end_date'].iloc[row])
        start_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], df['klinische_medicatie_medication_request_start_date'].iloc[row])
        end_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], df['klinische_medicatie_medication_request_end_date'].iloc[row])
        TEMP = False
        if TEMP:
            brandname = df['klinische_medicatie_medication_request_brand_name'].iloc[row]
            form = df['klinische_medicatie_medication_request_form'].iloc[row]
            qty = df['klinische_medicatie_medication_request_strength_value'].iloc[row]
            unit = df['klinische_medicatie_medication_request_strength_unit'].iloc[row]
        descr = df['klinische_medicatie_medication_request_description'].iloc[row]

        if not TEMP:
            data = {'pseudo_id': pseudo_id, 'start_time_adm': start_date_adm, 'end_time_adm': end_date_adm, 'description': descr}
        else:
            if symp_time:
                data = {'pseudo_id': pseudo_id, 'start_time_symp': start_date_symp, 'end_time_symp': end_date_symp, 'start_time_adm': start_date_adm,
                        'end_time_adm': end_date_adm, 'brandname': brandname, 'form': form, 'qty': qty, 'unit': unit, 'description': descr}
            else:
                data = {'pseudo_id': pseudo_id, 'start_time_adm': start_date_adm, 'end_time_adm': end_date_adm,
                        'brandname': brandname, 'form': form, 'qty': qty, 'unit': unit, 'description': descr}
        o_df = o_df.append(data, ignore_index=True)

    return o_df


def reformat_lab(df, patient_data, symp_time=True):
    print("\nProcessing lab data...")
    # Sort df
    df.sort_values(['pseudo_id', 'lab_measurement_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['pseudo_id', 'time_adm'] + (['time_symp'] if symp_time else []))
    o_row = -1
    pseudo_id = 0
    o_time = 0

    # Load keyfile for lab variables
    lab_key_df = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/lab_keyfile.xlsx")

    # Loop through df
    for row in range(len(df.index)):
        # Print progress information
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

        # Add row if (a) a new pseudo_id is registered, or (b) a new timepoint is registered - IF the feature is one of those matched between Haga and LUMC
        labcode = str(df.at[row, 'lab_measurement_id'])[13:]

        if labcode in list(lab_key_df[~pd.isna(lab_key_df['Feature Name'])]['LUMC Code']):
            if pseudo_id != df['pseudo_id'].iloc[row] or o_time != df['lab_measurement_start_date'].iloc[row]:
                o_row += 1
                pseudo_id = df['pseudo_id'].iloc[row]
                o_time = df['lab_measurement_start_date'].iloc[row]

            feature = lab_key_df.at[lab_key_df[lab_key_df['LUMC Code'] == labcode].index[0], 'Feature Name']
            value = df['lab_measurement_test_value_numeric'].iloc[row]

            # Add new column to output dataframe if feature has not yet been made
            if feature not in o_df.columns:
                o_df[feature] = np.nan

            # Populate row in output dataframe
            o_df.at[o_row, 'pseudo_id'] = pseudo_id
            o_df.at[o_row, feature] = value
            # Make absolute time point relative to first symptoms day
            if symp_time:
                time_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], o_time)
                o_df.at[o_row, 'time_symp'] = time_symp
            # Make absolute time point relative to admission day
            time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], o_time)
            o_df.at[o_row, 'time_adm'] = time_adm

    return o_df


def reformat_admission(df, patient_data, symp_time=True):
    print("\nProcessing admission data...")
    # Sort df
    df.sort_values(['pseudo_id', 'opname_delen_admission_parts_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    if symp_time:
        o_df = pd.DataFrame(columns=['pseudo_id', 'start_time_symp', 'end_time_symp', 'start_date_adm', 'end_date_adm', 'amdission_location', 'amdission_specialism', 'provider_position'])
    else:
        o_df = pd.DataFrame(columns=['pseudo_id', 'start_date_adm', 'end_date_adm', 'amdission_location', 'amdission_specialism', 'provider_position'])

    # Loop through df
    for row in range(len(df.index)):
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")
        pseudo_id = df['pseudo_id'].iloc[row]
        if symp_time:
            start_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], df.at[row, 'opname_delen_admission_parts_start_date'])
            end_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], df.at[row, 'opname_delen_admission_parts_end_date'])
        start_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], df.at[row, 'opname_delen_admission_parts_start_date'])
        end_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], df.at[row, 'opname_delen_admission_parts_end_date'])
        adm_loc = df.at[row, 'opname_delen_admission_parts_tags']
        adm_spec = df.at[row, 'opname_delen_admission_parts_specialism']
        provider_pos = df.at[row, 'opname_delen_admission_parts_care_provider_name']
        if symp_time:
            data = {'pseudo_id': pseudo_id, 'start_time_symp': start_date_symp, 'end_time_symp': end_date_symp, 'start_time_adm': start_date_adm,
                    'end_time_adm': end_date_adm, 'amdission_location': adm_loc, 'amdission_specialism': adm_spec, 'provider_position': provider_pos}
        else:
            data = {'pseudo_id': pseudo_id, 'start_time_adm': start_date_adm, 'end_time_adm': end_date_adm,
                    'amdission_location': adm_loc, 'amdission_specialism': adm_spec, 'provider_position': provider_pos}
        o_df = o_df.append(data, ignore_index=True)

    return o_df


def reformat_overview(df, patient_data, symp_time=True):
    print("\nProcessing overview data...")
    # Sort df
    df.sort_values(['pseudo_id', 'opname_delen_admission_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    if symp_time:
        cols = ['pseudo_id', 'start_time_symp', 'end_time_symp', 'start_time_adm', 'end_time_adm' 'age', 'sex', 'discharge_destination', 'diagnosis',
                'meds_count', 'lab_count', 'vitals_count', 'vms_count', 'complaints_count', 'q10_count', 'q11_count']
    else:
        cols = ['pseudo_id', 'start_time_adm', 'end_time_adm' 'age', 'sex', 'discharge_destination', 'diagnosis',
                'meds_count', 'lab_count', 'vitals_count', 'vms_count', 'complaints_count', 'q10_count', 'q11_count']
    o_df = pd.DataFrame(columns=cols)

    # Loop through df
    for row in range(len(df.index)):
        sd, ed = None, None  # Start- and end date
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")
        pseudo_id = df['pseudo_id'].iloc[row]

        if pseudo_id in patient_data.index:
            for col in ['opname_delen_admission_start_date', 'diagnose_report_start_date', 'diagnose_diagnosis_start_date']:
                if not pd.isna(df.at[row, col]):
                    sd = df.at[row, col]
                    break
            for col in ['opname_delen_admission_end_date', 'diagnose_diagnosis_end_date']:
                if not pd.isna(df.at[row, col]):
                    ed = df.at[row, col]
                    break

            if sd is not None and ed is not None and pseudo_id != '707E5F5BF0B40CD6D1CC9DF23A0D03E2B8927E4B':
                if symp_time:
                    start_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], sd)
                    end_date_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], ed)
                start_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], sd)
                end_date_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], ed)
                age = df.at[row, 'age']
                sex = df.at[row, 'gender']
                discharge_dest = df.at[row, 'opname_delen_admission_discharge_destination']
                diagnosis = df.at[row, 'diagnose_answer_label']
                meds_count = df.at[row, 'klinische_medicatie_collection_count']
                lab_count = df.at[row, 'lab_collection_count']
                vitals_count = df.at[row, 'vitale_metingen_collection_count']
                vms_count = df.at[row, 'vms_scores_collection_count']
                complaints_count = df.at[row, 'klachten_collection_count']
                q10_count = df.at[row, 'q_10_collection_count']
                q11_count = df.at[row, 'q_11_collection_count']

                if symp_time:
                    data = {'pseudo_id': pseudo_id, 'start_time_symp': start_date_symp, 'end_time_symp': end_date_symp, 'start_time_adm': start_date_adm,
                            'end_time_adm': end_date_adm, 'age': age, 'sex': sex, 'discharge_destination': discharge_dest, 'diagnosis': diagnosis,
                            'meds_count': meds_count, 'lab_count': lab_count, 'vitals_count': vitals_count, 'vms_count': vms_count,
                            'complaints_count': complaints_count, 'q10_count': q10_count, 'q11_count': q11_count}
                else:
                    data = {'pseudo_id': pseudo_id, 'start_time_adm': start_date_adm,'end_time_adm': end_date_adm,
                            'age': age, 'sex': sex, 'discharge_destination': discharge_dest, 'diagnosis': diagnosis,
                            'meds_count': meds_count, 'lab_count': lab_count, 'vitals_count': vitals_count, 'vms_count': vms_count,
                            'complaints_count': complaints_count, 'q10_count': q10_count, 'q11_count': q11_count}
                o_df = o_df.append(data, ignore_index=True)

    return o_df


def reformat_q10(df, patient_data):
    return None


def reformat_q11(df, patient_data):
    return None


def reformat_vms(df, patient_data, symp_time=True):
    print("\nProcessing vms data...")
    # Sort df
    df.sort_values(['pseudo_id', 'vms_scores_form_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['pseudo_id', 'time_adm'] + (['time_symp'] if symp_time else []))
    o_row = - 1
    pseudo_id = 0
    o_time = 0

    # Loop through df
    for row in range(len(df.index)):
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

        if pseudo_id != df['pseudo_id'].iloc[row] or o_time != df['vms_scores_form_start_date'].iloc[row]:
            o_row += 1
            pseudo_id = df['pseudo_id'].iloc[row]
            o_time = df['vms_scores_form_start_date'].iloc[row]

        descr = df['vms_scores_form_description'].iloc[row]
        question = df['vms_scores_form_entries_question'].iloc[row]
        value = None
        for col in ['vms_scores_form_entries_value_text', 'vms_scores_form_entries_value_multiple_choice',
                    'vms_scores_form_entries_value_number', 'vms_scores_form_entries_value_date']:
            if not pd.isna(df.at[row, col]):
                value = df.at[row, col]
                break
        time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], o_time)
        feature = f"{descr} - {question}"

        if feature not in o_df.columns:
            o_df[feature] = ''

        o_df.at[o_row, 'pseudo_id'] = pseudo_id
        o_df.at[o_row, feature] = str(value)
        if symp_time:
            time_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], o_time)
            o_df.at[o_row, 'time'] = time_symp
        o_df.at[o_row, 'time_adm'] = time_adm

    return o_df


def reformat_vitals(df, patient_data, symp_time=True):
    print("\nProcessing vitals data...")
    # Sort df
    df.sort_values(['pseudo_id', 'vitale_metingen_observation_start_date'], ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['pseudo_id', 'time_adm'] + (['time_symp'] if symp_time else []))
    o_row = - 1
    pseudo_id = 0
    o_time = 0

    # Load keyfile for lab variables
    vitals_key_df = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/vitals_keyfile.xlsx")

    # Loop through df
    for row in range(len(df.index)):
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

        if pseudo_id != df['pseudo_id'].iloc[row] or o_time != df['vitale_metingen_observation_start_date'].iloc[row]:
            o_row += 1
            pseudo_id = df['pseudo_id'].iloc[row]
            o_time = df['vitale_metingen_observation_start_date'].iloc[row]

        descr = df['vitale_metingen_observation_description'].iloc[row]

        feature = vitals_key_df.at[vitals_key_df[vitals_key_df['LUMC Description'] == descr].index[0], 'Feature Name']
        value = df['vitale_metingen_observation_test_value_numeric'].iloc[row]

        if feature not in o_df.columns:
            o_df[feature] = np.nan

        o_df.at[o_row, 'pseudo_id'] = pseudo_id
        o_df.at[o_row, feature] = value
        if symp_time:
            time_symp = abs_to_rel_date(patient_data['symptoms_date'].loc[pseudo_id], o_time)
            o_df.at[o_row, 'time'] = time_symp
        time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], o_time)
        o_df.at[o_row, 'time_adm'] = time_adm

    return o_df


# ARGUMENTS #
complaints_data = "covid-studie-rachel-lumc-klachten.csv"
meds_data = "covid-studie-rachel-lumc-klinische-medicatie.csv"
lab_data = "covid-studie-rachel-lumc-lab.csv"
admission_data = "covid-studie-rachel-lumc-opname-delen-admission-parts.csv"
overview_data = "covid-studie-rachel-lumc-overview.csv"
q10_data = "covid-studie-rachel-lumc-q-10.csv"
q11_data = "covid-studie-rachel-lumc-q-11.csv"
vitals_data = "covid-studie-rachel-lumc-vitale-metingen.csv"
vms_data = "covid-studie-rachel-lumc-vms-scores.csv"
mews_data = "covid-studie-rachel-lumc-MEWS-form-entries.csv"

patient_data = "Stats/Patient_Statistics_LUMC.csv"
#############


# Load all raw datasets
complaints_data, meds_data, lab_data, admission_data, overview_data, q10_data, q11_data, vitals_data, vms_data = load_data(complaints_data, meds_data, lab_data, admission_data,
                                                                                                                           overview_data, q10_data, q11_data, vitals_data, vms_data, mews_data)

# Load processed patient data, used for admission dates
patient_data = pd.read_csv(patient_data, delimiter=',', quotechar='"', index_col='pseudo_id')

# Process + save all individual data files
# pro_complaints = reformat_complaints(complaints_data, patient_data, symp_time=False)
# save_file(pro_complaints, 'Processed_Data/complaints_processed.csv', index=False)

# pro_meds = reformat_meds(meds_data, patient_data, symp_time=False)
# save_file(pro_meds, 'Processed_Data/medication_processed.csv', index=False)

# pro_admission = reformat_admission(admission_data, patient_data, symp_time=False)
# save_file(pro_admission, 'Processed_Data/admission_processed.csv', index=False)

# pro_overview = reformat_overview(overview_data, patient_data, symp_time=False)
# save_file(pro_overview, 'Processed_Data/overview_processed.csv', index=False)

# pro_q10 = reformat_q10(q10_data, patient_data, symp_time=False)
# pro_q10.to_csv('Processed_Data/q10_processed.csv', index=False)

# pro_q11 = reformat_q11(q11_data, patient_data, symp_time=False)
# pro_q11.to_csv('Processed_Data/q11_processed.csv', index=False)

# pro_vitals = reformat_vitals(vitals_data, patient_data, symp_time=False)
# save_file(pro_vitals, 'Processed_Data/vitals_processed.csv', index=False)

pro_lab = reformat_lab(lab_data, patient_data, symp_time=False)
save_file(pro_lab, 'Processed_Data/lab_processed.csv', index=False)

# pro_vms = reformat_vms(vms_data, patient_data, symp_time=False)
# save_file(pro_vms, 'Processed_Data/vms_processed.csv', index=False)

print("Done.")








