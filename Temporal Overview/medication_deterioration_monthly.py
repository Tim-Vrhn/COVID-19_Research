from matplotlib import pyplot as plt
from datetime import datetime as dt, timedelta
from Toolkit import *
import pandas as pd

"""
Plot a bar chart for the following:
1. Per month, the fraction of patients hospitalised that deteriorates (ICU + Death)
2. Per month, the fraction of patients hospitalised that receive dexamethasone, remdesivir, and hydroxychloroquine
"""


# Load data
haga_path = "/exports/reum/tverheijen/Haga_Data/02-02"
lumc_path = "/exports/reum/tverheijen/LUMC_Data/20210330"

haga_patients_df = exclude(f"{haga_path}/Stats/Patient_Statistics_Haga.csv", f"{haga_path}/Stats/Patient_Statistics_Haga.csv", symptom_onset=False)
lumc_patients_df = exclude(f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", symptom_onset=False)
patient_df = pd.concat([haga_patients_df, lumc_patients_df])
patient_df.set_index('pseudo_id', inplace=True)
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
patient_df['died_date'] = pd.to_datetime(patient_df['died_date'])
patient_df['admitted_to_IC'] = pd.to_datetime(patient_df['admitted_to_IC'])
haga_meds_df = exclude(f"{haga_path}/Processed_Data/medication_processed.csv", f"{haga_path}/Stats/Patient_Statistics_Haga.csv", symptom_onset=False)
lumc_meds_df = exclude(f"{lumc_path}/Processed_Data/medication_processed.csv", f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", symptom_onset=False)
meds_df = pd.concat([haga_meds_df, lumc_meds_df])
meds_df = meds_df[~pd.isna(meds_df['end_time_adm'])]
meds_df['start_time_adm'] = meds_df.apply(lambda row: patient_df.at[row['pseudo_id'], 'admitted_date'] + timedelta(hours=row['start_time_adm']), axis=1)
meds_df['end_time_adm'] = meds_df.apply(lambda row: patient_df.at[row['pseudo_id'], 'admitted_date'] + timedelta(hours=row['end_time_adm']), axis=1)

# Time range for every month
timestamps = pd.date_range(dt(2020, 3, 1), dt(2021, 2, 1), freq='MS')

# Set data lists
deteriorate = []
dexa = []
rem = []
hydro = []
chloro = []
toc = []

# Loop through dates
for i, time in enumerate(timestamps):
	next_time = timestamps[i + 1] if i < len(timestamps) - 1 else dt(2099, 1, 1)

	# Filter out patients
	hospitalised_patients = patient_df[(patient_df['admitted_date'] >= time) &
									   (patient_df['admitted_date'] <= next_time)].index
	if len(hospitalised_patients) > 0:
		# Number of patients that deteriorate (died patients, ICU patients, and then unique in both)
		deteriorated_patients = list(patient_df[(patient_df.index.isin(hospitalised_patients)) &
												(patient_df['died'] == 1) &
												(patient_df['died_date'] >= time) & (patient_df['died_date'] <= next_time)].index)
		deteriorated_patients.extend(list(patient_df[(patient_df.index.isin(hospitalised_patients)) &
													 (patient_df['admitted_to_IC'] >= time) & (patient_df['admitted_to_IC'] <= next_time)].index))
		deteriorated_patients = len(set(list(deteriorated_patients)))
		deteriorate.append(deteriorated_patients / len(hospitalised_patients) * 100)

		# Number of patients that received medication
		dexa.append(len(set(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
									(meds_df['description'].str.startswith('DEXAMETHASON')) &
									(meds_df['start_time_adm'] >= time) &
									(meds_df['start_time_adm'] <= next_time)]['pseudo_id'])) / len(hospitalised_patients) * 100)
		rem.append(len(set(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
								   (meds_df['description'].str.startswith('REMDESIVIR')) &
								   (meds_df['start_time_adm'] >= time) &
								   (meds_df['start_time_adm'] <= next_time)]['pseudo_id'])) / len(hospitalised_patients) * 100)
		hydro.append(len(set(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
									 (meds_df['description'].str.startswith('HYDROXYCHLOROQUINE')) &
									 (meds_df['start_time_adm'] >= time) &
									 (meds_df['start_time_adm'] <= next_time)]['pseudo_id'])) / len(hospitalised_patients) * 100)
		chloro.append(len(set(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
									  (meds_df['description'].str.startswith('CHLOROQUINE')) &
									  (meds_df['start_time_adm'] >= time) &
									  (meds_df['start_time_adm'] <= next_time)]['pseudo_id'])) / len(hospitalised_patients) * 100)
		chloro_general = [i for i in list(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
									  (meds_df['description'].str.startswith('CHLOROQUINE')) &
									  (meds_df['start_time_adm'] >= time) &
									  (meds_df['start_time_adm'] <= next_time)]['pseudo_id']) if i in list(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
									  (meds_df['description'].str.startswith('HYDROXYCHLOROQUINE')) &
									  (meds_df['start_time_adm'] >= time) &
									  (meds_df['start_time_adm'] <= next_time)]['pseudo_id'])]
		print(len(chloro_general), "patients received both CQ and HCQ between", time, "and", next_time)

		toc.append(len(set(meds_df[(meds_df['pseudo_id'].isin(hospitalised_patients)) &
								   (meds_df['description'].str.startswith('TOCILIZUMAB')) &
								   (meds_df['start_time_adm'] >= time) &
								   (meds_df['start_time_adm'] <= next_time)]['pseudo_id'])) / len(hospitalised_patients) * 100)
	else:
		deteriorate.append(0)
		dexa.append(0)
		rem.append(0)
		hydro.append(0)
		chloro.append(0)
		toc.append(0)
exit()
# Set plot parameters
fontsize = 16
plt.rcParams.update({'font.size': fontsize})
labels = ['Deterioration', 'Chloroquine', 'Hydroxychloroquine', 'Remdesivir', 'Dexamethasone', 'Tocilizumab']
colours = ['#fcab10'] + colour_palette()
width = 1 / len(labels[:-1])
x_ticks = timestamps
x_ticks_pos = x_tick_pos = [[i + width * j for i in range(len(x_ticks))] for j in range(len(labels) - 1)]

# Plot
fig, ax = plt.subplots(figsize=(16, 5))
for i, data in enumerate([chloro, hydro, rem, dexa, toc]):
	ax.bar(x_tick_pos[i], data, align='center', width=width, label=labels[i + 1], color=colours[i + 1], zorder=2)
ax.plot(x_tick_pos[2], deteriorate, color=colours[0], label=labels[0], zorder=1)
ax.fill_between(x_tick_pos[2], deteriorate, color=colours[0], zorder=0)

# Change labels
ax.set_ylabel("% of hospitalised patients")
ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=6)
ax.set_xticks([i - 0.5 * width for i in range(len(x_ticks))])
ax.set_xticklabels([i.strftime("%m/%Y") for i in x_ticks], rotation=45, ha='right')
ax.grid(c="#d9d9d9", axis='x')
plt.ylim(0, 100)
plt.tight_layout()
save_file(fig, "/exports/reum/tverheijen/Combined/Figures/Monthly_Medication_and_Deterioration.png")
