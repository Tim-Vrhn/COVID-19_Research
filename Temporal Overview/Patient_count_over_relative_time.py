from matplotlib import pyplot as plt
from datetime import timedelta
from Toolkit import *
from math import ceil
import pandas as pd

"""
1. Plot three line graphs showing how many patients were hospitalised on the ward at times relative to 1st day of symptoms

2. Plot three line graphs showing how many patients were hospitalised on the ward at times relative to 1st day of admission
"""

# Read data
haga_path = "/exports/reum/tverheijen/Haga_Data/02-02"
lumc_path = "/exports/reum/tverheijen/LUMC_Data/20210330"
haga_patients_df = exclude(f"{haga_path}/Stats/Patient_Statistics_Haga.csv", f"{haga_path}/Stats/Patient_Statistics_Haga.csv", symptom_onset=False)
lumc_patients_df = exclude(f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", symptom_onset=False)
patients_df = pd.concat([haga_patients_df, lumc_patients_df], ignore_index=True)
patients_df['admitted_date'] = pd.to_datetime(patients_df['admitted_date'])
patients_df['discharge_date'] = pd.to_datetime(patients_df['discharge_date'])
patients_df['admitted_to_IC'] = pd.to_datetime(patients_df['admitted_to_IC'])
patients_df['discharged_from_IC'] = pd.to_datetime(patients_df['discharged_from_IC'])

# Get labels, colours
clr_discharged, lbl_discharged = colour_scheme('discharged')
clr_died, lbl_died = colour_scheme('died')
clr_icu, lbl_icu = colour_scheme('ICU')

if 1 == 0:
	# Set up x-axis
	days = max((patients_df['discharge_date'] - patients_df['symptoms_date']).max(),
			   (patients_df['discharge_date'] - patients_df['admitted_date']).max()).days
	x_tick_pos = [i for i in range(days)]

	# Gather counts over time
	discharged_counts = []
	icu_counts = []
	died_counts = []
	group_dict = {'group_discharged': discharged_counts, 'group_ICU': icu_counts, 'group_died': died_counts}
	for i in x_tick_pos:
		# Check, for each group, the number of patients still present
		for group in group_dict:
			group_dict[group].append(len(patients_df[(patients_df[group] == 1) &
													 (patients_df['admitted_date'] <= (patients_df['symptoms_date'] + timedelta(days=i))) &
													 ((patients_df['symptoms_date'] + timedelta(days=i)) <=
													  (patients_df['discharge_date'] if group != 'group_ICU' else patients_df['admitted_to_IC']))
													 ].index))

	# Plot
	fig, ax = plt.subplots(figsize=(16, 6))
	for i, group in enumerate(group_dict):
		ax.plot(x_tick_pos, group_dict[group], color=[clr_discharged, clr_icu, clr_died][i], label=[lbl_discharged, lbl_icu, lbl_died][i])

	# labels etc.
	fontsize = 22
	plt.rcParams.update({'font.size': fontsize})
	ax.legend()
	ax.set_title("Number of patients hospitalised on ward after symptom onset")
	ax.set_ylabel("Patients hospitalised on ward", fontsize=22)
	ax.set_xlabel("Time after first symptomps (days)")
	ax.spines['left'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.grid(c="#ebebeb")
	fig.tight_layout()

	save_file(fig, "Stats/Patient_count_over_relative_time.png")

# Set up x-axis
days = ceil((patients_df['hospitalisation_time'] / 24).max())
x_tick_pos = [i for i in range(days)]

# Gather counts over time
group_dict = {'group_discharged': [], 'group_ICU': [], 'died': []}
for i in x_tick_pos:
	# Check, for each group, the number of patients still present before discharge (or death) / ICU admission (if ICU patient)
	for group in group_dict:
		count = 0
		if group != 'group_ICU':
			for _, row in patients_df[patients_df[group] == 1].iterrows():
				if row['admitted_date'] + timedelta(days=i) <= (row['discharge_date'] if pd.isna(row['admitted_to_IC']) else row['admitted_to_IC']):
					count += 1
		else:
			for _, row in patients_df[(patients_df[group] == 1) & (patients_df['died'] == 0)].iterrows():
				if row['admitted_date'] + timedelta(days=i) <= row['admitted_to_IC']:
					count += 1

		group_dict[group].append(count)

	# Once all patients are admitted to the ICU / discharged / deceased, stop counting (it will be just 0s from now)
	if sum([group_dict[group][-1] for group in group_dict]) == 0:
		break

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
for i, group in enumerate(group_dict):
	ax.plot([x_tick_pos[i] for i in range(len(group_dict[group]))], group_dict[group], color=[clr_discharged, clr_icu, clr_died][i],
			label=[lbl_discharged, lbl_icu, lbl_died][i])

# labels etc.
ax.legend()
ax.set_title("Number of patients hospitalised on ward after initial hospitalisation")
ax.set_ylabel("Patients hospitalised on ward")
ax.set_xlabel("Time after initial hospitalisation (days)")
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(c="#ebebeb")
fig.tight_layout()

save_file(fig, "Figures/Patient_count_over_time.png")
