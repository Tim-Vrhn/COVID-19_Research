from Toolkit import *

"""
(1)
Some patients are admitted to the ER which doesn't show up as the admitted date
Reset admission to ER admission in patients_df file

(2)
Also sets "No ICU policy" for LUMC/Haga patients based on manually generated file

(3)
Adds column to keyfile (patient_df) indicating whether a patient was admitted from the SEH straight to the ICU (this is regarded as a direct ICU admission)
"""

# Parameters
data = 'LUMC'
noicupolicy_df = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/No_ICU_policy.xlsx")

######## 1

# Load data
patient_df = pd.read_csv(f"Stats/Patient_Statistics_{data}.csv")
patient_df.set_index('pseudo_id', inplace=True)
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
patient_df['admitted_to_IC'] = pd.to_datetime(patient_df['admitted_to_IC'])
patient_df['discharged_from_IC'] = pd.to_datetime(patient_df['discharged_from_IC'])
patient_df['died_date'] = pd.to_datetime(patient_df['died_date'])
lab_df = pd.read_csv(f"covid-studie-rachel-{'lumc-' if data == 'LUMC' else ''}lab.csv")
lab_df['lab_measurement_start_date'] = pd.to_datetime(lab_df['lab_measurement_start_date'])
vitals_df = pd.read_csv(f"covid-studie-rachel-{'lumc-' if data == 'LUMC' else ''}vitale-metingen.csv")
vitals_df['vitale_metingen_observation_start_date'] = pd.to_datetime(vitals_df['vitale_metingen_observation_start_date'])
meds_df = pd.read_csv(f"covid-studie-rachel-{'lumc-' if data == 'LUMC' else ''}klinische-medicatie.csv")
meds_df['klinische_medicatie_medication_request_start_date'] = pd.to_datetime(meds_df['klinische_medicatie_medication_request_start_date'])
meds_df['start_date_time'] = [meds_df.at[i, 'klinische_medicatie_medication_request_start_date'].time() for i in meds_df.index]

# Loop through patients
if False:
    for pseudo_id in patient_df.index:
        adm_date = patient_df.loc[pseudo_id, "admitted_date"] if not pd.isna(patient_df.loc[pseudo_id, "admitted_date"]) else datetime.datetime(2999, 1, 1)

        first_vitals_record = vitals_df[(vitals_df['pseudo_id'] == pseudo_id) & (~pd.isna(vitals_df['vitale_metingen_observation_start_date']))]['vitale_metingen_observation_start_date'].min() \
            if pseudo_id in list(vitals_df['pseudo_id']) else datetime.datetime(2999, 1, 1)
        first_lab_record = lab_df[(lab_df['pseudo_id'] == pseudo_id) & (~pd.isna(lab_df['lab_measurement_start_date']))]['lab_measurement_start_date'].min() \
            if pseudo_id in list(lab_df['pseudo_id']) else datetime.datetime(2999, 1, 1)
        first_med_record = meds_df[(meds_df['pseudo_id'] == pseudo_id) & (~pd.isna(meds_df['klinische_medicatie_medication_request_start_date'])) &
                                   (meds_df['start_date_time'] != datetime.time())]['klinische_medicatie_medication_request_start_date'].min() \
            if pseudo_id in list(meds_df['pseudo_id']) else datetime.datetime(2999, 1, 1)
        first_record = min(first_vitals_record, first_lab_record, first_med_record)

        if first_record < adm_date:
            # print(f"Patient {pseudo_id} has an earlier record than the first admission date ({first_record} vs. {adm_date})")

            patient_df.loc[pseudo_id, 'admitted_date'] = min(first_record, adm_date)
            patient_df.loc[pseudo_id, 'admitted_loc'] = 'SEH'
        if not pd.isna(patient_df.loc[pseudo_id, 'admitted_date']):
            if patient_df.at[pseudo_id, 'group_ICU'] == 1:
                patient_df.loc[pseudo_id, 'admitted_to_IC_after_hours'] = int(round((patient_df.loc[pseudo_id, 'admitted_to_IC'] - patient_df.loc[pseudo_id, 'admitted_date']).total_seconds() / 3600))
                if not pd.isna(patient_df.loc[pseudo_id, 'discharged_from_IC']):
                    patient_df.loc[pseudo_id, 'discharged_from_IC_after_hours'] = int(round((patient_df.loc[pseudo_id, 'discharged_from_IC'] - patient_df.loc[pseudo_id, 'admitted_date']).total_seconds() / 3600))
            if patient_df.at[pseudo_id, 'died'] == 1:
                patient_df.loc[pseudo_id, 'died_after_hours'] = int(round((patient_df.loc[pseudo_id, 'died_date'] - patient_df.loc[pseudo_id, 'admitted_date']).total_seconds() / 3600))


######## 2

for row in range(len(noicupolicy_df.index)):
    col = 0 if data == 'Haga' else 1
    if pd.isna(noicupolicy_df.iloc[row, col]):
        break
    pseudo_id = noicupolicy_df.iloc[row, col]

    if pseudo_id in patient_df.index:
        patient_df.loc[pseudo_id, 'no_IC'] = 1
        patient_df.loc[pseudo_id, 'NRNB'] = 1


######## 3
admission_df = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/covid-studie-rachel-lumc-opname-delen-admission-parts.csv")
admission_df['opname_delen_admission_parts_start_date'] = pd.to_datetime(admission_df['opname_delen_admission_parts_start_date'])
admission_df.sort_values(['pseudo_id', 'opname_delen_admission_parts_start_date'], inplace=True, ascending=[True, True])
admission_df = admission_df[['pseudo_id', 'opname_delen_admission_parts_start_date', 'opname_delen_admission_parts_tags', 'opname_delen_admission_parts_department']]

if 'SEH_to_ICU' not in patient_df.columns:
    icu_bln = []
    for pseudo_id in patient_df.index:
        if pseudo_id in patient_df[(patient_df['admitted_loc'] == 'SEH') & (~pd.isna(patient_df['admitted_to_IC']))].index:
            adm_date = patient_df.at[pseudo_id, 'admitted_date']
            admission_patient_df = admission_df[(admission_df['pseudo_id'] == pseudo_id) & (admission_df['opname_delen_admission_parts_start_date'] > adm_date) &
                                                (admission_df['opname_delen_admission_parts_department'] != 'Centrum Eerste Hulp')]

            if admission_patient_df.iloc[0].loc['opname_delen_admission_parts_tags'] == 'IC voor Volwassenen':
                icu_bln.append(1)
            else:
                icu_bln.append(0)
        else:
            icu_bln.append(0)

    patient_df.insert(10, 'SEH_to_ICU', icu_bln)


######## 4
if 'SEH_only' not in patient_df.columns:
    seh_bln = []
    for pseudo_id in patient_df.index:
        if len(admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique()) == 1:
            print("________________________", pseudo_id)
            print(admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique())
            print(len(admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique()))
            print(admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique()[0])
        if len(admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique()) == 1 and \
                admission_df[admission_df['pseudo_id'] == pseudo_id]['opname_delen_admission_parts_department'].unique()[0] == 'Centrum Eerste Hulp':
            seh_bln.append(1)
        else:
            seh_bln.append(0)
    patient_df.insert(13, 'SEH_only', seh_bln)

save_file(patient_df, f"Stats/Patient_Statistics_{data}.csv")
