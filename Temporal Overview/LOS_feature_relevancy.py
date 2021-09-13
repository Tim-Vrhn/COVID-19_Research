from Toolkit import *

"""
for BOTH HOSPITALS

Generates a table (.csv file) with various features and their values for patients with different short LOS' 
Columns: Up to 1 day, 0-2 days, 0-3 days, etc.
Rows: features such as Ox. supply, medication, demographics, outcome, group
"""

# Load patient data
haga_path = '/exports/reum/tverheijen/Haga_Data/02-02/'
lumc_path = '/exports/reum/tverheijen/LUMC_Data/20210330/'

excl_filters = {'treatment_limit': False, 'dis_other_hospital': False, 'symptom_onset': False, 'icu': False, 'seh_to_icu': False}

haga_patient_df = exclude(f"{haga_path}Stats/Patient_Statistics_Haga.csv", f"{haga_path}Stats/Patient_Statistics_Haga.csv", **excl_filters)
haga_patient_df.set_index('pseudo_id', inplace=True)
lumc_patient_df = exclude(f"{lumc_path}Stats/Patient_Statistics_LUMC.csv", f"{lumc_path}Stats/Patient_Statistics_LUMC.csv", **excl_filters)
# Load vitals data
haga_all_vitals_df = exclude(f"{haga_path}Processed_Data/vitals_processed.csv", f"{haga_path}Stats/Patient_Statistics_Haga.csv", **excl_filters)
lumc_all_vitals_df = exclude(f"{lumc_path}Processed_Data/vitals_processed.csv", f"{lumc_path}Stats/Patient_Statistics_LUMC.csv", **excl_filters)
# Load Lab data
haga_all_lab_df = exclude(f"{haga_path}Processed_Data/lab_processed.csv", f"{haga_path}Stats/Patient_Statistics_Haga.csv", **excl_filters)
lumc_all_lab_df = exclude(f"{lumc_path}Processed_Data/lab_processed.csv", f"{lumc_path}Stats/Patient_Statistics_LUMC.csv", **excl_filters)
# Load meds data
haga_all_meds_df = exclude(f"{haga_path}Processed_Data/medication_processed.csv", f"{haga_path}Stats/Patient_Statistics_Haga.csv", **excl_filters)
lumc_all_meds_df = exclude(f"{lumc_path}Processed_Data/medication_processed.csv", f"{lumc_path}Stats/Patient_Statistics_LUMC.csv", **excl_filters)

# Join data from both hospitals
patient_df = pd.concat([haga_patient_df, lumc_patient_df])
all_vitals_df = pd.concat([haga_all_vitals_df, lumc_all_vitals_df])
all_lab_df = pd.concat([haga_all_lab_df, lumc_all_lab_df])
all_meds_df = pd.concat([haga_all_meds_df, lumc_all_meds_df])

# Define different LOS groups (max. nr. of days)
LOS_groups = [1, 2, 3, 4, 5, float('inf')]


# Generate o_df
rows = ['Total patients', 'Age', 'Female (%)', 'Directly adm. to ICU (%)', 'Adm. to ICU (%)', 'Diseased (%)', 'Treat. Limit. (%)',
        'Origin: usual place of residency (%)', 'Origin: care home (%)', 'Origin: institution (other) (%)',
        'Dest.: usual place of residency (%)', 'Dest.: care home (%)', 'Dest.: institution (other) (%)', 'Dest.: other hospital (%)', 'Dest.: deceased (%)',
        'Patients with supp. O2 (%)', 'Mean supp. O2 (L/min)',
        'Corona Labscore', # 'MEWS',
        'Tocilizumab (%)', 'Dexamethasone (%)', 'Remdesivir (%)']
o_df = pd.DataFrame(index=rows, columns=[f'0-{i}' if i < 99 else f'0-{LOS_groups[-2]}' for i in LOS_groups])

#
# Populate o_df
#
LOS_groups = [i * 24 for i in LOS_groups]

for i, los in enumerate(LOS_groups):
    patients = patient_df[patient_df['hospitalisation_time'] <= los].index
    tot_patients = len(patients)

    # Filter out relevant patients for vitals and meds data
    meds_df = all_meds_df[all_meds_df['pseudo_id'].isin(patients)]
    vitals_df = all_vitals_df[all_vitals_df['pseudo_id'].isin(patients)]
    lab_df = all_lab_df[all_lab_df['pseudo_id'].isin(patients)]

    # Total patients
    o_df.iloc[0, i] = tot_patients

    # Age
    o_df.loc['Age'].iloc[i] = round(patient_df[patient_df['hospitalisation_time'] <= los]['age'].mean(), 1)

    # Sex
    o_df.loc['Female (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) & (patient_df['sex'] == 'F')].index)
                                            / tot_patients) * 100, 1)

    # Adm. to ICU
    o_df.loc['Adm. to ICU (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) & (~pd.isna(patient_df['admitted_to_IC']))].index)
                                                 / tot_patients) * 100, 1)

    # Dir. adm. to ICU
    o_df.loc['Directly adm. to ICU (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                         ((patient_df['admitted_loc'].isin(['IC', 'IC voor Volwassenen']))|
                                                                          (patient_df['SEH_to_ICU'] == 1))].index)
                                                      / tot_patients) * 100, 1)

    # Diseased
    o_df.loc['Diseased (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) & (patient_df['died'] == 1)].index)
                                              / tot_patients) * 100, 1)

    # Treat. Limit
    o_df.loc['Treat. Limit. (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) & (patient_df['NRNB'] == 1)].index)
                                                  / tot_patients) * 100, 1)

    # Origins
    o_df.loc['Origin: usual place of residency (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                                     (patient_df['admission_origin'] == 'Eigen woonomgeving')].index)
                                                                / tot_patients) * 100, 1)
    o_df.loc['Origin: care home (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                      (patient_df['admission_origin'].isin(['Verpleeg-/verzorgingshuis', 'Instelling vr verpleging verz']))].index)
                                                                       / tot_patients) * 100, 1)
    o_df.loc['Origin: institution (other) (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                                (patient_df['admission_origin'].isin(
                                                                                    ['Instelling (anders)', 'Overige instellingen', 'GGZ instelling', 'Instelling voor revalidatie']))].index)
                                                                 / tot_patients) * 100, 1)

    # Destinations
    o_df.loc['Dest.: usual place of residency (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                                    (patient_df['discharge_dest'].isin(['Eigen woonomgeving', 'Naar huis', 'Eigen woonomgeving met zorg' ]))].index)
                                                              / tot_patients) * 100, 1)
    o_df.loc['Dest.: care home (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                     (patient_df['discharge_dest'].isin(['Verpleeg- / verzorgingshuis', 'Naar instelling vr verpleging']))].index)
                                                                       / tot_patients) * 100, 1)
    o_df.loc['Dest.: institution (other) (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                               (patient_df['discharge_dest'].isin(['Instelling (psychiatrisch)',
                                                                                                                   'Naar instelling vr revalidatie',
                                                                                                                   'Instelling (anders)', 'Instelling (revalidatie)',
                                                                                                                   'Naar instelling vr revalidatie', 'Hospice',
                                                                                                                   'Naar GGZ instelling', 'Naar overige instelling']))].index)
                                                      / tot_patients) * 100, 1)
    o_df.loc['Dest.: other hospital (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                          (patient_df['discharge_dest'].str.contains('ziekenhuis', case=False))].index)
                                                            / tot_patients) * 100, 1)
    o_df.loc['Dest.: deceased (%)'].iloc[i] = round((len(patient_df[(patient_df['hospitalisation_time'] <= los) &
                                                                    (patient_df['discharge_dest'].str.startswith('Overleden'))].index)
                                                     / tot_patients) * 100, 1)
    # Supplementary O2
    o_df.loc['Patients with supp. O2 (%)'].iloc[i] = round((len(vitals_df[(vitals_df['time_adm'] <= los) & (~pd.isna(vitals_df['Supplemental Oxygen (L/min)']) & (vitals_df['Supplemental Oxygen (L/min)'] != 0))]
                                                                ['pseudo_id'].unique()) / tot_patients) * 100, 1)

    o_df.loc['Mean supp. O2 (L/min)'].iloc[i] = round(vitals_df[(vitals_df['time_adm'] <= los) & (~pd.isna(vitals_df['Supplemental Oxygen (L/min)']) & (vitals_df['Supplemental Oxygen (L/min)'] != 0))]
                                                      ['Supplemental Oxygen (L/min)'].mean(), 1)

    # Scores
    # o_df.loc['MEWS'].iloc[i] = round(vitals_df[(vitals_df['time_adm'] <= los) & (~pd.isna(vitals_df['MEWS']) & (vitals_df['MEWS'] != 0))]
                                    # ['MEWS'].mean(), 1)
    o_df.loc['Corona Labscore'].iloc[i] = round(lab_df[(lab_df['time_adm'] <= los) & (~pd.isna(lab_df['Corona Labscore 1']) & (lab_df['Corona Labscore 1'] != 0))]
                                                  ['Corona Labscore 1'].mean(), 1)
    # o_df.loc['Corona Labscore 2'].iloc[i] = round(lab_df[(lab_df['time_adm'] <= los) & (~pd.isna(lab_df['Corona Labscore 2']) & (lab_df['Corona Labscore 2'] != 0))]
                                                  # ['Corona Labscore 2'].mean(), 1)

    # Medication
    o_df.loc['Dexamethasone (%)'].iloc[i] = round((len(meds_df[(meds_df['start_time_adm'] <= los) & (meds_df['description'].str.lower().str.startswith('dexamethason'))]['pseudo_id'].unique())
                                                   / tot_patients) * 100, 1)
    o_df.loc['Tocilizumab (%)'].iloc[i] = round((len(meds_df[(meds_df['start_time_adm'] <= los) & (meds_df['description'].str.lower().str.startswith('tocilizumab'))]['pseudo_id'].unique())
                                                 / tot_patients) * 100, 1)
    o_df.loc['Remdesivir (%)'].iloc[i] = round((len(meds_df[(meds_df['start_time_adm'] <= los) & (meds_df['description'].str.lower().str.startswith('remdesivir'))]['pseudo_id'].unique())
                                                / tot_patients) * 100, 1)

print(o_df)

save_file(o_df, "CSV/LOS_feature_relevancy_BothHospitals.csv")
