import pandas as pd
import numpy as np
from datetime import timedelta
from Toolkit import save_file

def baseline_urine(pseudo_id, data, patient_df):
	data = data[(data['pseudo_id'] == pseudo_id) &
				(data['urine_toevoeging_rachel_measurement_start_date'] <= patient_df.at[pseudo_id, 'admitted_date'] + timedelta(hours=24))]
	baseline_value = data['urine_toevoeging_rachel_measurement_test_value_numeric'].iloc[0] if len(data.index) > 0 else np.nan
	return baseline_value


# Load data
patient_df = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/Stats/Patient_Statistics_LUMC.csv", delimiter=',')
patient_df.set_index('pseudo_id', inplace=True)
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
vitals_df = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-vitale-metingen-bij-opname.csv", delimiter=',', quotechar='"')
vitals_df = vitals_df[vitals_df['pseudo_id'].isin(patient_df.index)]
vitals_df['vitale_metingen_bij_opname_observation_start_date'] = pd.to_datetime(vitals_df['vitale_metingen_bij_opname_observation_start_date'])
vitals_df.sort_values(['pseudo_id', 'vitale_metingen_bij_opname_observation_description', 'vitale_metingen_bij_opname_observation_start_date']
					  , ignore_index=True, inplace=True)
vitals_key = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/vitals_keyfile.xlsx")
vitals_key = vitals_key[~pd.isna(vitals_key['Feature Name'])]

# Make baseline df
baseline_vitals = pd.DataFrame(index=vitals_df['pseudo_id'].unique())

vitalsdescr = tuple(vitals_key['LUMC Description'])
vitalsdescr_dict = {}
for descr in vitalsdescr:
	vitalsdescr_dict[descr] = vitals_key[vitals_key['LUMC Description'] == descr]['Feature Name']

for pseudo_id in baseline_vitals.index:
	v_df = vitals_df[(vitals_df['pseudo_id'] == pseudo_id) &
					 (vitals_df['vitale_metingen_bij_opname_observation_start_date'] <= patient_df.at[pseudo_id, 'admitted_date'] + timedelta(hours=24))]
	for feature in v_df['vitale_metingen_bij_opname_observation_description'].unique():
		if len(v_df[v_df['vitale_metingen_bij_opname_observation_description'] == feature].index) > 0:
			baseline_vitals.loc[pseudo_id, vitalsdescr_dict[feature].iloc[0]] = \
				v_df[v_df['vitale_metingen_bij_opname_observation_description'] == feature]['vitale_metingen_bij_opname_observation_test_value_numeric'].iloc[0]

baseline_vitals.index.name = 'pseudo_id'
print(baseline_vitals)
save_file(baseline_vitals, "/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/baseline_vitals_LUMC.csv")




