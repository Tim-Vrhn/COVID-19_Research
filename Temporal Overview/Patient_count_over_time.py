import pandas as pd
from matplotlib import pyplot as plt
from Toolkit import exclude, colour_scheme, save_file
from datetime import datetime
"""
Plots a bar chart 
Shows, per day, how many patients were hospitalised in 2020 - early 2021
"""
haga_path = "/exports/reum/tverheijen/Haga_Data/02-02"
lumc_path = "/exports/reum/tverheijen/LUMC_Data/20210330"
haga_patients_df = exclude(f"{haga_path}/Stats/Patient_Statistics_Haga.csv", f"{haga_path}/Stats/Patient_Statistics_Haga.csv", symptom_onset=False)
lumc_patients_df = exclude(f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", symptom_onset=False)
df = pd.concat([haga_patients_df, lumc_patients_df])
df.sort_values(['discharge_date'], ignore_index=True, inplace=True)
df['admitted_date'] = pd.to_datetime(df['admitted_date'])
df['discharge_date'] = pd.to_datetime(df['discharge_date'])
# Get labels, colours
clr_discharged, lbl_discharged = colour_scheme('discharged')
clr_died, lbl_died = colour_scheme('died')
clr_icu, lbl_icu = colour_scheme('ICU')
# Patient counts for every day starting from first to last date
first_date = df.loc[0, 'discharge_date'].replace(hour=0, minute=0, second=0)
last_date = df.loc[df.index[-1], 'discharge_date'].replace(hour=0, minute=0, second=0)
timestamps = pd.date_range(datetime(2020, 1, 1), datetime(2021, 5, 1), freq='1D')
patient_counts = [len(df[(df['admitted_date'] <= time) & (df['discharge_date'] >= time)]['pseudo_id'].unique()) for time in timestamps]
ICU_counts = [len(df[(df['admitted_date'] <= time) & (df['discharge_date'] >= time) & (df['group_ICU'] == 1) & (df['died'] == 0)]['pseudo_id'].unique()) for time in timestamps]
died_counts = [len(df[(df['admitted_date'] <= time) & (df['discharge_date'] >= time) & (df['died'] == 1)]['pseudo_id'].unique()) for time in timestamps]
# Plot bar plot
x_tick_pos = [i for i in range(0, len(patient_counts))]
# Plot barplot
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x_tick_pos, patient_counts, align='center', label=lbl_discharged, width=1.0, color=clr_discharged)
ax.bar(x_tick_pos, ICU_counts, align='center', width=1.0, bottom=died_counts, label=lbl_icu, color=clr_icu)
ax.bar(x_tick_pos, died_counts, align='center', width=1.0, label=lbl_died, color=clr_died)
ax.legend()
ax.set_xticks([i for i in range(0, len(patient_counts), 7)])
ax.set_xticklabels([timestamps[t].strftime("%d/%m/%Y") if i % 2 == 0 else '' for i, t in enumerate(range(0, len(timestamps), 7))], rotation=45, ha='right')
ax.set_ylabel("Patients")
plt.tight_layout()
save_file(plt, "Figures/Patient_Count_2020-21_Both.png")