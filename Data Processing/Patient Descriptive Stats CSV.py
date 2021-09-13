from datetime import datetime
from Toolkit import exclude
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import numpy as np
import re


"""
This script produces Patient_Statistics_Haga.csv, a file with info on all patients in the diff data files combined.
Contains information such as demographics, death date, admission data, NRNB information, etc.
"""

# ARGUMENTS #
save_output_df = True
output_filename = "Stats/Patient_Statistics_LUMC.csv"
admission_filename = "covid-studie-rachel-lumc-opname-delen-admission-parts.csv"
vms_scores_filename = "covid-studie-rachel-lumc-vms-scores.csv"
overview_filename = "covid-studie-rachel-lumc-overview.csv"

cutoff_dates = [datetime(2020, 10, 15), datetime(2020, 8, 1), datetime(2020, 7, 28)]
#############


def populate(patient_id, overview_data, admission_data, nrnb_data, patientstats_df):
    if patient_id in overview_data.index:
        if pd.isna(overview_data.at[patient_id, 'opname_delen_admission_start_date']):
            return patientstats_df

    # Add patient ID
    patientstats_df = patientstats_df.append(pd.Series({'NRNB': 0, 'no_resuscitate': 0, 'no_intubate': 0, 'no_IC': 0, 'age': 0, 'died': 0,
                                                        'admitted_date': datetime(1970, 1, 1), 'admitted_to_IC': datetime(1970, 1, 1),
                                                        'discharged_from_IC': datetime(1970, 1, 1)},
                                                       name=patient_id))
    # Set columns to right types
    patientstats_df['sex'] = patientstats_df['sex'].astype('str')

    if patient_id in list(nrnb_data['pseudo_id']):
        # Populate NRNB
        patient_nrnb = nrnb_data[nrnb_data['pseudo_id'] == patient_id]
        patient_nrnb = patient_nrnb[patient_nrnb['vms_scores_form_entries_question'].isin(['Reanimeren', 'Beademen', 'IC opname'])]
        patient_nrnb.set_index('vms_scores_form_entries_question', inplace=True)
        nrnb_dict = {'Reanimeren': 'no_resuscitate', 'Beademen': 'no_intubate', 'IC opname': 'no_IC'}
        for key in nrnb_dict.keys():
            if key in list(patient_nrnb.index):
                if str(patient_nrnb.at[key, 'vms_scores_form_entries_value_text']).strip() in ['ja', 'nee']:
                    patientstats_df.at[patient_id, nrnb_dict[key]] = int(patient_nrnb.at[key, 'vms_scores_form_entries_value_text']
                                                                         .strip().replace('ja', '0').replace('nee', '1'))
                    if patientstats_df.at[patient_id, nrnb_dict[key]] == 1:
                        patientstats_df.at[patient_id, 'NRNB'] = 1

    if patient_id in list(admission_data.index):
        # Populate admitted_date, discharge_date, admitted_loc
        if len(admission_data.loc[patient_id].index) > 1 and admission_data.loc[patient_id].index[0] != 'opname_delen_admission_parts_start_date':
            for row in range(len(admission_data.loc[patient_id].index)):
                if patientstats_df.at[patient_id, 'admitted_date'] == datetime(1970, 1, 1):
                    patientstats_df.at[patient_id, 'admitted_date'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date']
                    patientstats_df.at[patient_id, 'admitted_loc'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_tags'] \
                        if admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_department'] != 'Centrum Eerste Hulp' else 'SEH'
                else:
                    if admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date'] < patientstats_df.at[patient_id, 'admitted_date']:
                        patientstats_df.at[patient_id, 'admitted_date'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date']
                        patientstats_df.at[patient_id, 'admitted_loc'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_tags'] \
                            if admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_department'] != 'Centrum Eerste Hulp' else 'SEH'
        else:
            patientstats_df.at[patient_id, 'admitted_date'] = admission_data.at[patient_id, 'opname_delen_admission_parts_start_date']
            patientstats_df.at[patient_id, 'admitted_loc'] = admission_data.at[patient_id, 'opname_delen_admission_parts_tags']
        patientstats_df.at[patient_id, 'discharge_date'] = overview_data.at[patient_id, 'opname_delen_admission_end_date']
        # Populate admitted_before_cutoff
        for cutoff_date in cutoff_dates:
            if patientstats_df.at[patient_id, 'admitted_date'] < cutoff_date:
                patientstats_df.at[patient_id, f'admitted_before_{cutoff_date.day}-{cutoff_date.month}'] = 1
            else:
                patientstats_df.at[patient_id, f'admitted_before_{cutoff_date.day}-{cutoff_date.month}'] = 0
        # Populate admitted_to_IC, discharged_from_IC
        if len(admission_data.loc[patient_id].index) > 1 and admission_data.loc[patient_id].index[0] != 'opname_delen_admission_parts_start_date':
            for row in range(len(admission_data.loc[patient_id].index)):
                nrow = row + 1 if row < len(admission_data.loc[patient_id].index) - 1 else len(admission_data.loc[patient_id].index) - 1
                if 'IC' in str(admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_tags']):
                    if patientstats_df.at[patient_id, 'admitted_to_IC'] == datetime(1970, 1, 1):
                        patientstats_df.at[patient_id, 'admitted_to_IC'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date']
                    else:
                        if admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date'] < patientstats_df.at[patient_id, 'admitted_to_IC']:
                            patientstats_df.at[patient_id, 'admitted_to_IC'] = admission_data.loc[patient_id].iloc[row]['opname_delen_admission_parts_start_date']
                    if 'IC' not in str(admission_data.loc[patient_id].iloc[nrow]['opname_delen_admission_parts_tags']) and \
                            patientstats_df.at[patient_id, 'discharged_from_IC'] == datetime(1970, 1, 1):
                        patientstats_df.at[patient_id, 'discharged_from_IC'] = admission_data.loc[patient_id].iloc[nrow]['opname_delen_admission_parts_start_date']
        else:
            if 'IC' in str(admission_data.at[patient_id, 'opname_delen_admission_parts_tags']):
                patientstats_df.at[patient_id, 'admitted_to_IC'] = admission_data.at[patient_id, 'opname_delen_admission_parts_start_date']
                patientstats_df.at[patient_id, 'discharged_from_IC'] = admission_data.at[patient_id, 'opname_delen_admission_parts_end_date']

        # Populate admitted_to_IC_after_hours, discharged_from_IC_after_hours
        if patientstats_df.at[patient_id, 'admitted_to_IC'] > datetime(1970, 1, 1):
            admitted_to_ic_after_hours = patientstats_df.at[patient_id, 'admitted_to_IC'] - patientstats_df.at[patient_id, 'admitted_date']
            patientstats_df.at[patient_id, 'admitted_to_IC_after_hours'] = int(round(admitted_to_ic_after_hours.total_seconds() / 3600))
        if patientstats_df.at[patient_id, 'discharged_from_IC'] > datetime(1970, 1, 1):
            admitted_to_ic_after_hours = patientstats_df.at[patient_id, 'discharged_from_IC'] - patientstats_df.at[patient_id, 'admitted_date']
            patientstats_df.at[patient_id, 'discharged_from_IC_after_hours'] = int(round(admitted_to_ic_after_hours.total_seconds() / 3600))

    if patient_id in list(overview_data.index):
        # Populate sex
        patientstats_df.at[patient_id, 'sex'] = str(overview_data.at[patient_id, 'gender'])
        # Populate age
        patientstats_df.at[patient_id, 'age'] = int(overview_data.at[patient_id, 'age'])
        # Populate diagnosis
        patientstats_df.at[patient_id, 'diagnosis'] = overview_data.at[patient_id, 'diagnose_answer_label']
        # Populate died, died_date, died_after_hours
        if re.search("overleden", str(overview_data.at[patient_id, 'opname_delen_admission_discharge_destination'])
                .replace(" ", "").lower()) is not None:
            patientstats_df.at[patient_id, 'died'] = 1
            patientstats_df.at[patient_id, 'died_date'] = overview_data.at[patient_id, 'opname_delen_admission_end_date']
            died_after_hours = patientstats_df.at[patient_id, 'died_date'] - patientstats_df.at[patient_id, 'admitted_date']
            patientstats_df.at[patient_id, 'died_after_hours'] = int(round(died_after_hours.total_seconds() / 3600))
        # Populate admission_origin
        patientstats_df.at[patient_id, 'admission_origin'] = overview_data.at[patient_id, 'opname_delen_admission_admission_source']
        # Populate discharge_dest
        patientstats_df.at[patient_id, 'discharge_dest'] = overview_data.at[patient_id, 'opname_delen_admission_discharge_destination']
        # Populate hospitalisation_time
        hospitalisation_time_hours = overview_data.at[patient_id, 'opname_delen_admission_end_date'] - patientstats_df.at[patient_id, 'admitted_date']
        patientstats_df.at[patient_id, 'hospitalisation_time'] = int(round(hospitalisation_time_hours.total_seconds() / 3600))
    # Populate group_discharged, group_ICU, group_died
    patientstats_df.at[patient_id, 'group_discharged'] = 1 if patientstats_df.at[patient_id, 'died'] == 0 and pd.isna(patientstats_df.at[
                                                                                                                          patient_id, 'admitted_to_IC_after_hours']) else 0
    patientstats_df.at[patient_id, 'group_ICU'] = 1 if not pd.isna(patientstats_df.at[patient_id, 'admitted_to_IC_after_hours']) else 0
    if not pd.isna(patientstats_df.at[patient_id, 'admitted_to_IC']):
        died_after_IC = True if patientstats_df.at[patient_id, 'admitted_to_IC_after_hours'] < patientstats_df.at[patient_id, 'died_after_hours'] else False
    else:
        died_after_IC = False
    patientstats_df.at[patient_id, 'group_died'] = 1 if patientstats_df.at[patient_id, 'died'] == 1 and not died_after_IC else 0

    return patientstats_df

# Load data
admission_data = pd.read_csv(admission_filename, delimiter=',', quotechar='"')[
    ['pseudo_id', 'opname_delen_admission_parts_start_date', 'opname_delen_admission_parts_end_date', 'opname_delen_admission_parts_tags', 'opname_delen_admission_parts_department']]
nrnb_data = pd.read_csv(vms_scores_filename, delimiter=',', quotechar='"')[
    ['pseudo_id', 'vms_scores_form_entries_question', 'vms_scores_form_entries_value_text']]
overview_data = pd.read_csv(overview_filename, delimiter=',', quotechar='"')[
    ['pseudo_id', 'gender', 'age', 'opname_delen_admission_discharge_destination', 'opname_delen_admission_start_date', 'opname_delen_admission_admission_source',
     'opname_delen_admission_end_date', 'diagnose_answer_label']]

# Make empty patient statistics df
col_list = ['pseudo_id', 'sex', 'age', 'diagnosis', 'admitted_date', 'discharge_date', 'admitted_loc', 'admission_origin', 'admitted_to_IC', 'admitted_to_IC_after_hours',
            'discharged_from_IC', 'discharged_from_IC_after_hours', 'discharge_dest', 'NRNB', 'died', 'died_date', 'died_after_hours', 'hospitalisation_time', 'group_discharged', 'group_ICU', 'group_died']
col_list[6:6] = [f"admitted_before_{d.day}-{d.month}" for d in cutoff_dates]
patientstats_df = pd.DataFrame(columns=col_list)

# Set patient ID as index
overview_data.set_index('pseudo_id', inplace=True)
patientstats_df.set_index('pseudo_id', inplace=True)

# Set data columns to datetime objects
admission_data['opname_delen_admission_parts_start_date'] = pd.to_datetime(admission_data['opname_delen_admission_parts_start_date'])
admission_data['opname_delen_admission_parts_end_date'] = pd.to_datetime(admission_data['opname_delen_admission_parts_end_date'])
overview_data['opname_delen_admission_end_date'] = pd.to_datetime(overview_data['opname_delen_admission_end_date'])
overview_data['opname_delen_admission_start_date'] = pd.to_datetime(overview_data['opname_delen_admission_start_date'])

# Sort admission data
admission_data.sort_values(['pseudo_id', 'opname_delen_admission_parts_start_date'], ignore_index=True, inplace=True)
admission_data.set_index('pseudo_id', inplace=True)

# Gather all available patient IDs in one list of unique IDs
patient_id_list = []

for i, data in enumerate([admission_data.index, nrnb_data['pseudo_id'], overview_data.index]):
    addendum = [i for i in list(data) if len(i) == 40]
    patient_id_list.extend(addendum)
patient_id_list = sorted(set(patient_id_list))

# Populate per patient ID
for i, patient_id in enumerate(patient_id_list):
    # Print progression
    if i % round(len(patient_id_list) / 5) == 0:
        print(f"Progress: patient {i} / {len(patient_id_list)} ({round(i / len(patient_id_list) * 100, 1)}%)")
    patientstats_df = populate(patient_id, overview_data, admission_data, nrnb_data, patientstats_df)

# Replace the placeholder datetime values with N/A
patientstats_df.replace(datetime(1970, 1, 1), np.nan, inplace=True)

# Save the data
if save_output_df:
    patientstats_df.to_csv(output_filename)
    print("Saved file:", output_filename)

#
#
# Generate descriptive statistics from generated data

columns = ['all', 'initially hospitalised in clinic (non-IC)', 'initially hospitalized in clinic (non-IC) no behandelbeperking',
           'initially hospitalized in IC', 'initially hospitalized in IC no behandelbeperking', 'behandelbeperking',
           'no resuscitation order', 'no intubation order', 'no IC order', 'no to all order']
rows = ['total', 'behandelbeperking (%)', 'IC admission (%)', 'deaths (%)', 'deaths (%) after IC admission',
        'average time to death (days) ', 'mean symptom duration at hospital admission', 'mean duration of hospitalization (days)',
        'PCR+ diagnosis (%)', 'CORADS 4 diagnosis (%)', 'CORADS 5 diagnosis (%)', 'CORADS 6 diagnosis (%)', 'COVID-19 code diagnosis (%)',
        'mean age', 'female (%)']
stats_total = pd.DataFrame(columns=columns)
stats_before_date = pd.DataFrame(columns=columns)
stats_after_date = pd.DataFrame(columns=columns)
for row in rows:
    stats_total = stats_total.append(pd.Series(name=row))
    stats_before_date = stats_before_date.append(pd.Series(name=row))
    stats_after_date = stats_after_date.append(pd.Series(name=row))

for cutoff_date in cutoff_dates:
    print(f"Saving data for cutoff date {cutoff_date.day}-{cutoff_date.month}-{cutoff_date.year}")
    for i, df in enumerate([patientstats_df,
                            patientstats_df[patientstats_df[f"admitted_before_{cutoff_date.day}-{cutoff_date.month}"] == 1],
                            patientstats_df[patientstats_df[f"admitted_before_{cutoff_date.day}-{cutoff_date.month}"] == 0]]):
        stats_df = [stats_total, stats_before_date, stats_after_date][i]

        non_IC = df[df['admitted_loc'] == 'Klinische opname']
        non_IC_no_NRNB = non_IC[non_IC['NRNB'] == 0]
        IC = df[df['admitted_loc'] == 'IC']
        IC_no_NRNB = IC[IC['NRNB'] == 0]
        NRNB = df[df['NRNB'] == 1]
        no_resuscitate = df[df['no_resuscitate'] == 1]
        no_intubate = df[df['no_intubate'] == 1]
        no_IC = df[df['no_IC'] == 1]
        no_all = df[(df['no_resuscitate'] == 1) & (df['no_intubate'] == 1) & (df['no_IC'] == 1)]

        for j, sub_df in enumerate([df, non_IC, non_IC_no_NRNB, IC, IC_no_NRNB, NRNB, no_resuscitate, no_intubate, no_IC, no_all]):
            total = len(sub_df.index)
            if total > 0:
                behandelbeperking = len(sub_df[sub_df['NRNB'] == 1].index)
                behandelbeperking_perc = round(behandelbeperking / total * 100, 1)
                IC_admissions = len(sub_df[pd.isna(sub_df['admitted_to_IC']) == 0].index)
                IC_admissions_perc = round(IC_admissions / total * 100, 1)
                number_of_deaths = len(sub_df[sub_df['died'] == 1].index)
                deaths_perc = round(number_of_deaths / total * 100, 1)
                deaths_after_IC = len(sub_df[pd.isna(sub_df['admitted_to_IC']) == 0][sub_df[pd.isna(sub_df['admitted_to_IC']) == 0]['died'] == 1].index)
                deaths_after_IC_perc = round(deaths_after_IC / IC_admissions * 100, 1)
                avg_time_to_death = round(sum([i / 24 for i in sub_df[sub_df['died'] == 1]['died_after_hours']]) / number_of_deaths, 1)
                avg_symptom_duration = "N/A"
                avg_hospit_duration = round(sum([i / 24 for i in sub_df['hospitalisation_time']]) / total, 1)
                PCR_diagnosis = len(sub_df[sub_df['diagnosis'] == 'Positieve PCR'].index)
                CORADS4_diagnosis = len(sub_df[sub_df['diagnosis'] == 'CORADS 4'].index)
                CORADS5_diagnosis = len(sub_df[sub_df['diagnosis'] == 'CORADS 5'].index)
                CORADS6_diagnosis = len(sub_df[sub_df['diagnosis'] == 'CORADS 6'].index)
                COVID19_diagnosis = len(sub_df[sub_df['diagnosis'] == 'Covid_code'].index)
                PCR_diagnosis_perc = round(PCR_diagnosis / total * 100, 1)
                CORADS4_diagnosis_perc = round(CORADS4_diagnosis / total * 100, 1)
                CORADS5_diagnosis_perc = round(CORADS5_diagnosis / total * 100, 1)
                CORADS6_diagnosis_perc = round(CORADS6_diagnosis / total * 100, 1)
                COVID19_diagnosis_perc = round(COVID19_diagnosis / total * 100, 1)
                avg_age = round(sum(list(sub_df['age'])) / total, 1)
                female_count = len(sub_df[sub_df['sex'] == 'F'].index)
                perc_female = round(female_count / total * 100, 1)

                stats_df[columns[j]] = [total, f"{behandelbeperking} ({behandelbeperking_perc})", f"{IC_admissions} ({IC_admissions_perc})",
                                        f"{number_of_deaths} ({deaths_perc})", f"{deaths_after_IC} ({deaths_after_IC_perc})", avg_time_to_death,
                                        avg_symptom_duration, avg_hospit_duration, f"{PCR_diagnosis} ({PCR_diagnosis_perc})", f"{CORADS4_diagnosis}"
                                        f" ({CORADS4_diagnosis_perc})", f"{CORADS5_diagnosis} ({CORADS5_diagnosis_perc})", f"{CORADS6_diagnosis} "
                                        f"({CORADS6_diagnosis_perc})", f"{COVID19_diagnosis} ({COVID19_diagnosis_perc})", avg_age,
                                        f"{female_count} ({perc_female})"]
            else:
                stats_df[columns[j]] = "N/A"
        stats_df.to_csv(f"Stats/Stats_{['all', f'before_{cutoff_date.day}-{cutoff_date.month}', f'after_{cutoff_date.day}-{cutoff_date.month}'][i]}.csv")
