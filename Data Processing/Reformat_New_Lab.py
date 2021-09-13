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
lab_df = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-overview.csv", delimiter=',', quotechar='"')
lab_df.set_index('pseudo_id', inplace=True)
lab_df = lab_df[lab_df.index.isin(patient_df.index)]
lab_key = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/lab_keyfile.xlsx")
lab_key = lab_key[(~pd.isna(lab_key['LUMC Code'])) & (~pd.isna(lab_key['Feature Name']))]
urine_df = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-urine-toevoeging-rachel.csv")
urine_df['urine_toevoeging_rachel_measurement_start_date'] = pd.to_datetime(urine_df['urine_toevoeging_rachel_measurement_start_date'])

# Make baseline DF
baseline_lab = pd.DataFrame(index=lab_df.index)

labcodes = tuple(lab_key[~pd.isna(lab_key['LUMC Code'])]['LUMC Code'])
labcodes_dict = {}
for labcode in labcodes:
	labcodes_dict[labcode] = lab_key[lab_key['LUMC Code'] == labcode]['Feature Name']

for i, col in enumerate(lab_df.columns):
	if col[-6:] == 'ent_id' and len(lab_df[~pd.isna(lab_df[col])].index) > 0 and lab_df[~pd.isna(lab_df[col])][col].iloc[0].endswith(labcodes):
		# Determine correct output column name
		for key in labcodes_dict:
			o_col = None
			if lab_df[~pd.isna(lab_df[col])][col].iloc[0].endswith(key):
				o_col = labcodes_dict[key].iloc[0]
				break
		baseline_lab[o_col] = lab_df.apply(lambda row: row[row.index[i + 3]], axis=1)

# Voeg urinemetingen totaal eiwit toe
baseline_lab['Total Protein UP (g/L)'] = baseline_lab.apply(lambda row: baseline_urine(row.name, urine_df, patient_df), axis=1)

save_file(baseline_lab, "/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/baseline_lab_LUMC.csv")




