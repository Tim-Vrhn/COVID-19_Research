import pandas as pd
import numpy as np
from os.path import exists
from Toolkit import save_file

def abs_to_rel_date(base_date, date):
	"""
	Converts absolute datetime object to relative time
	:param base_date: time 0
	:param date: absolute date + time (string) to be converted
	:return: float, hours after admission
	"""

	time_diff = round((pd.to_datetime(date) - pd.to_datetime(base_date)).total_seconds() / 3600, 1)
	return time_diff


def reformat_vitals(df, time_col, descr_col, val_col, patient_data, centre):
	print("\nProcessing vitals data...")
	# Sort df
	df.sort_values(['pseudo_id', time_col], ignore_index=True, inplace=True)
	# New empty output df
	o_df = pd.DataFrame(columns=['pseudo_id', 'time_adm'])
	o_row = - 1
	pseudo_id = 0
	o_time = 0

	# Load keyfile for lab variables
	vitals_key_df = pd.read_excel("/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/vitals_keyfile.xlsx")

	# Loop through df
	for row in range(len(df.index)):
		if row % round(len(df.index) / 8) == 0:
			print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

		if pseudo_id != df['pseudo_id'].iloc[row] or o_time != df[time_col].iloc[row]:
			o_row += 1
			pseudo_id = df['pseudo_id'].iloc[row]
			o_time = df[time_col].iloc[row]

		descr = df[descr_col].iloc[row]

		feature = vitals_key_df.at[vitals_key_df[vitals_key_df[f'{centre} Description'] == descr].index[0], 'Feature Name']
		value = df[val_col].iloc[row]

		if feature not in o_df.columns:
			o_df[feature] = np.nan

		o_df.at[o_row, 'pseudo_id'] = pseudo_id
		o_df.at[o_row, feature] = value
		time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], o_time)
		o_df.at[o_row, 'time_adm'] = time_adm

	return o_df


def reformat_lab(df, id_col, time_col, val_col, patient_data, centre):
	print("\nProcessing lab data...")
	# Sort df
	df.sort_values(['pseudo_id', time_col], ignore_index=True, inplace=True)
	# New empty output df
	o_df = pd.DataFrame(columns=['pseudo_id', 'time_adm'])
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
		labcode = str(df.at[row, id_col])[(-8 if centre == 'Haga' else 13):]

		if labcode in list(lab_key_df[~pd.isna(lab_key_df['Feature Name'])][f'{centre} Code']):
			if pseudo_id != df['pseudo_id'].iloc[row] or o_time != df[time_col].iloc[row]:
				o_row += 1
				pseudo_id = df['pseudo_id'].iloc[row]
				o_time = df[time_col].iloc[row]

			feature = lab_key_df.at[lab_key_df[lab_key_df[f'{centre} Code'] == labcode].index[0], 'Feature Name']
			value = df[val_col].iloc[row]

			# Add new column to output dataframe if feature has not yet been made
			if feature not in o_df.columns:
				o_df[feature] = np.nan

			# Populate row in output dataframe
			o_df.at[o_row, 'pseudo_id'] = pseudo_id
			o_df.at[o_row, feature] = value

			# Make absolute time point relative to admission day
			time_adm = abs_to_rel_date(patient_data['admitted_date'].loc[pseudo_id], o_time)
			o_df.at[o_row, 'time_adm'] = time_adm

	return o_df


"""PROCESS LUMC DATA"""
if False:
	# Load patient keyfile
	patient_df = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/Stats/Patient_Statistics_LUMC.csv', index_col='pseudo_id')

	# Load and Transform labdata
	if exists("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_tijdens.csv"):
		lab_tijdens_opname = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_tijdens.csv", index_col=0)
		lab_toevoeging = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_toevoeging.csv", index_col=0)
		lab_urine = pd.read_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_urine.csv", index_col=0)
	else:
		lab_tijdens_opname = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-labuitslagen-tijdens-opname.csv')
		lab_tijdens_opname = lab_tijdens_opname[lab_tijdens_opname['pseudo_id'].isin(patient_df.index)]
		lab_tijdens_opname['labuitslagen_tijdens_opname_measurement_start_date'] = pd.to_datetime(lab_tijdens_opname['labuitslagen_tijdens_opname_measurement_start_date'])
		lab_toevoeging = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-lab-tijdens-opname-toevoeging-r.csv')
		lab_toevoeging = lab_toevoeging[lab_toevoeging['pseudo_id'].isin(patient_df.index)]
		lab_toevoeging['lab_tijdens_opname_toevoeging_rachel_measurement_start_date'] = pd.to_datetime(lab_toevoeging['lab_tijdens_opname_toevoeging_rachel_measurement_start_date'])
		lab_urine = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-urine-toevoeging-rachel.csv')
		lab_urine = lab_urine[lab_urine['pseudo_id'].isin(patient_df.index)]
		lab_urine['urine_toevoeging_rachel_measurement_start_date'] = pd.to_datetime(lab_urine['urine_toevoeging_rachel_measurement_start_date'])

		lab_tijdens_opname = reformat_lab(df=lab_tijdens_opname, patient_data=patient_df, id_col='labuitslagen_tijdens_opname_measurement_id',
										  time_col='labuitslagen_tijdens_opname_measurement_start_date', val_col='labuitslagen_tijdens_opname_measurement_test_value_numeric',
										  centre='LUMC')
		lab_toevoeging = reformat_lab(df=lab_toevoeging, patient_data=patient_df, id_col='lab_tijdens_opname_toevoeging_rachel_measurement_id',
									  time_col='lab_tijdens_opname_toevoeging_rachel_measurement_start_date',
									  val_col='lab_tijdens_opname_toevoeging_rachel_measurement_test_value_numeric',
									  centre='LUMC')
		lab_urine = reformat_lab(df=lab_urine, patient_data=patient_df, id_col='urine_toevoeging_rachel_measurement_id',
								 time_col='urine_toevoeging_rachel_measurement_start_date', val_col='urine_toevoeging_rachel_measurement_test_value_numeric',
								 centre='LUMC')

		lab_tijdens_opname.to_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_tijdens.csv")
		lab_toevoeging.to_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_toevoeging.csv")
		lab_urine.to_csv("/exports/reum/tverheijen/LUMC_Data/20210330/TEMP_lab_urine.csv")

	# Load and Transform vitalsdata
	vitals_tijdens_opname = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-vitale-metingen-tijdens-opname.csv')
	vitals_tijdens_opname = vitals_tijdens_opname[vitals_tijdens_opname['pseudo_id'].isin(patient_df.index)]
	vitals_tijdens_opname['vitale_metingen_tijdens_opname_observation_start_date'] = pd.to_datetime(vitals_tijdens_opname['vitale_metingen_tijdens_opname_observation_start_date'])
	o2_supp = pd.read_csv('/exports/reum/tverheijen/LUMC_Data/20210330/isaric-final-copy-from-haga-zuurstoftoediening-tijdens-opna.csv')
	o2_supp = o2_supp[o2_supp['pseudo_id'].isin(patient_df.index)]
	o2_supp['isaric-final-copy-from-haga-vitale-metingen-bij-opname.csv'] = pd.to_datetime(o2_supp['zuurstoftoediening_tijdens_opname_observation_start_date'])

	vitals_tijdens_opname = reformat_vitals(df=vitals_tijdens_opname, patient_data=patient_df, descr_col='vitale_metingen_tijdens_opname_observation_description',
											time_col='vitale_metingen_tijdens_opname_observation_start_date',
											val_col='vitale_metingen_tijdens_opname_observation_test_value_numeric', centre='LUMC')
	o2_supp = reformat_vitals(df=o2_supp, patient_data=patient_df, descr_col='zuurstoftoediening_tijdens_opname_observation_description',
							  time_col='zuurstoftoediening_tijdens_opname_observation_start_date',
							  val_col='zuurstoftoediening_tijdens_opname_observation_test_value_numeric', centre='LUMC')

	# Merge lab files
	lab_tijdens_opname = pd.merge(lab_tijdens_opname, lab_toevoeging, on=['pseudo_id', 'time_adm'], how='outer')
	print(lab_tijdens_opname.shape)
	lab_tijdens_opname = pd.merge(lab_tijdens_opname, lab_urine, on=['pseudo_id', 'time_adm'], how='outer')
	for col in [i for i in lab_tijdens_opname.columns if i.endswith('_x')]:
		ycol = col[:-2] + "_y"
		lab_tijdens_opname[col] = lab_tijdens_opname.apply(lambda row: row[ycol] if pd.isna(row[col]) else row[col], axis=1)
		lab_tijdens_opname.rename(columns={col: col[:-2]}, inplace=True)
		lab_tijdens_opname.drop(columns=ycol, inplace=True)
	print(lab_tijdens_opname)
	save_file(lab_tijdens_opname, '/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/lab_processed.csv', index=False)

	# Merge vitals files
	vitals_tijdens_opname = pd.merge(vitals_tijdens_opname, o2_supp, on=['pseudo_id', 'time_adm'], how='outer')
	print(vitals_tijdens_opname)
	save_file(vitals_tijdens_opname, '/exports/reum/tverheijen/LUMC_Data/20210330/Processed_Data/vitals_processed.csv', index=False)

"""PROCESS HAGA DATA"""
# Load patient keyfile
patient_df = pd.read_csv('/exports/reum/tverheijen/Haga_Data/02-02/Stats/Patient_Statistics_Haga.csv', index_col='pseudo_id')

# Load and Transform labdata
if exists("/exports/reum/tverheijen/Haga_Data/02-02/TEMP_lab_tijdens.csv"):
	lab_tijdens_opname = pd.read_csv("/exports/reum/tverheijen/Haga_Data/02-02/TEMP_lab_tijdens.csv", index_col=0)
	lab_toevoeging = pd.read_csv("/exports/reum/tverheijen/Haga_Data/02-02/TEMP_lab_toevoeging.csv", index_col=0)
else:
	lab_tijdens_opname = pd.read_csv('/exports/reum/tverheijen/Haga_Data/02-02/isaric-final-labuitslagen-tijdens-opname.csv')
	lab_tijdens_opname = lab_tijdens_opname[lab_tijdens_opname['pseudo_id'].isin(patient_df.index)]
	lab_tijdens_opname['labuitslagen_tijdens_opname_measurement_start_date'] = pd.to_datetime(lab_tijdens_opname['labuitslagen_tijdens_opname_measurement_start_date'])
	lab_toevoeging = pd.read_csv('/exports/reum/tverheijen/Haga_Data/02-02/isaric-final-lab-tijdens-opname-toevoeging-r.csv')
	lab_toevoeging = lab_toevoeging[lab_toevoeging['pseudo_id'].isin(patient_df.index)]
	lab_toevoeging['lab_tijdens_opname_toevoeging_rachel_measurement_start_date'] = pd.to_datetime(lab_toevoeging['lab_tijdens_opname_toevoeging_rachel_measurement_start_date'])

	lab_tijdens_opname = reformat_lab(df=lab_tijdens_opname, patient_data=patient_df, id_col='labuitslagen_tijdens_opname_measurement_id',
									  time_col='labuitslagen_tijdens_opname_measurement_start_date', val_col='labuitslagen_tijdens_opname_measurement_test_value_numeric',
									  centre='Haga')
	lab_toevoeging = reformat_lab(df=lab_toevoeging, patient_data=patient_df, id_col='lab_tijdens_opname_toevoeging_rachel_measurement_id',
								  time_col='lab_tijdens_opname_toevoeging_rachel_measurement_start_date',
								  val_col='lab_tijdens_opname_toevoeging_rachel_measurement_test_value_numeric',
								  centre='Haga')
	lab_tijdens_opname.to_csv("/exports/reum/tverheijen/Haga_Data/02-02/TEMP_lab_tijdens.csv")
	lab_toevoeging.to_csv("/exports/reum/tverheijen/Haga_Data/02-02/TEMP_lab_toevoeging.csv")
print(len(lab_tijdens_opname.index), len(lab_toevoeging.index))

# Load and Transform vitalsdata
vitals_tijdens_opname = pd.read_csv('/exports/reum/tverheijen/Haga_Data/02-02/isaric-final-vitale-metingen-tijdens-opname.csv')
vitals_tijdens_opname = vitals_tijdens_opname[vitals_tijdens_opname['pseudo_id'].isin(patient_df.index)]
vitals_tijdens_opname['vitale_metingen_tijdens_opname_observation_start_date'] = pd.to_datetime(
	vitals_tijdens_opname['vitale_metingen_tijdens_opname_observation_start_date'])
o2_supp = pd.read_csv('/exports/reum/tverheijen/Haga_Data/02-02/isaric-final-zuurstoftoediening-tijdens-opna.csv')
o2_supp = o2_supp[o2_supp['pseudo_id'].isin(patient_df.index)]
o2_supp['isaric-final-vitale-metingen-bij-opname.csv'] = pd.to_datetime(o2_supp['zuurstoftoediening_tijdens_opname_observation_start_date'])

vitals_tijdens_opname = reformat_vitals(df=vitals_tijdens_opname, patient_data=patient_df, descr_col='vitale_metingen_tijdens_opname_observation_description',
										time_col='vitale_metingen_tijdens_opname_observation_start_date',
										val_col='vitale_metingen_tijdens_opname_observation_test_value_numeric', centre='Haga')
o2_supp = reformat_vitals(df=o2_supp, patient_data=patient_df, descr_col='zuurstoftoediening_tijdens_opname_observation_description',
						  time_col='zuurstoftoediening_tijdens_opname_observation_start_date',
						  val_col='zuurstoftoediening_tijdens_opname_observation_test_value_numeric', centre='Haga')

# Merge lab files
lab_tijdens_opname = pd.merge(lab_tijdens_opname, lab_toevoeging, on=['pseudo_id', 'time_adm'], how='outer')
for col in [i for i in lab_tijdens_opname.columns if i.endswith('_x')]:
	print(col, "<-------------------------------")
	ycol = col[:-2] + "_y"
	lab_tijdens_opname[col] = lab_tijdens_opname.apply(lambda row: row[ycol] if pd.isna(row[col]) else row[col], axis=1)
	lab_tijdens_opname.rename(columns={col: col[:-2]}, inplace=True)
	lab_tijdens_opname.drop(columns=ycol, inplace=True)
print(lab_tijdens_opname)
save_file(lab_tijdens_opname, '/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/lab_processed.csv', index=False)

# Merge vitals files
vitals_tijdens_opname = pd.merge(vitals_tijdens_opname, o2_supp, on=['pseudo_id', 'time_adm'], how='outer')
print(vitals_tijdens_opname)
save_file(vitals_tijdens_opname, '/exports/reum/tverheijen/Haga_Data/02-02/Processed_Data/vitals_processed.csv', index=False)