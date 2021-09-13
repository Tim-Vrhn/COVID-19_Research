from Toolkit import *
from matplotlib import pyplot as plt
import seaborn as sns

"""
1. Print how many patients (in each group) had a LOS of <3 days
2. Monthly LOS for the three groups
3. LOS for the three groups, for various age categories
4. LOS for the three groups & sex
"""

# Load patient data
o_path = "/exports/reum/tverheijen/Combined/Figures"
haga_path = "/exports/reum/tverheijen/Haga_Data/02-02"
lumc_path = "/exports/reum/tverheijen/LUMC_Data/20210330"
haga_patients_df = exclude(f"{haga_path}/Stats/Patient_Statistics_Haga.csv", f"{haga_path}/Stats/Patient_Statistics_Haga.csv", symptom_onset=False)
lumc_patients_df = exclude(f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", f"{lumc_path}/Stats/Patient_Statistics_LUMC.csv", symptom_onset=False)

patient_df = pd.concat([haga_patients_df, lumc_patients_df])
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
patient_df.set_index('pseudo_id', inplace=True)

# Get group colours and labels
clr_discharged, lbl_discharged = colour_scheme('discharged')
clr_died, lbl_died = colour_scheme('died')
clr_icu, lbl_icu = colour_scheme('ICU')
hue_order = [lbl_discharged, lbl_icu, lbl_died]

#######
# 1. Print how many patients (in each group) had a LOS of <3 days
print(f"Discharged (no ICU) patients with LOS of <3 days: {len(patient_df[(patient_df['group_discharged'] == 1) & (patient_df['hospitalisation_time'] < 72)].index)}")
print(f"Discharged after ICU patients with LOS of <3 days: {len(patient_df[(patient_df['group_ICU'] == 1) & (patient_df['died'] == 0) & (patient_df['hospitalisation_time'] < 72)].index)}")
print(f"Deceased patients with LOS of <3 days: {len(patient_df[(patient_df['died'] == 1) & (patient_df['hospitalisation_time'] < 72)].index)}")


#######
# 2. Monthly LOS for the three groups

df = pd.DataFrame(columns=['Admission Month', 'Group', 'LOS (days)'], index=patient_df.index)

# Loop through patients
for pseudo_id in patient_df.index:
    month = patient_df.at[pseudo_id, 'admitted_date'].strftime("'%y %m")
    group = lbl_discharged if patient_df.at[pseudo_id, 'group_discharged'] == 1 else lbl_died if patient_df.at[pseudo_id, 'died'] == 1 else lbl_icu
    los = patient_df.at[pseudo_id, 'hospitalisation_time'] / 24

    df.loc[pseudo_id] = [month, group, los]

df.sort_values('Admission Month', inplace=True)
print(df.groupby(by=['Admission Month', 'Group']).size().unstack(fill_value=0))

# Plot boxplots
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x="Admission Month", y="LOS (days)", hue="Group", data=df, palette=[clr_discharged, clr_icu, clr_died], hue_order=hue_order)
ax.set_ylabel('LOS (days)', fontsize=13)
ax.set_xlabel('Admission Month', fontsize=13)
# plt.ylim(-5, 100)
plt.tight_layout()
save_file(fig, f"{o_path}/LOS_Monthly.png")


#######
# 3. LOS for the three groups, for various age categories

df = pd.DataFrame(columns=['Age Group', 'Group', 'LOS (days)'], index=patient_df.index)

# Loop through patients
age_groups = {(0, 18): '(0-18)', (18, 30): '(18-30)', (30, 40): '(30-40)', (40, 50): '(40-50)', (50, 60): '(50-60)', (60, 70): '(60-70)', (70, float('inf')): '(70+)'}
for pseudo_id in patient_df.index:
    for i in age_groups:
        if i[0] <= patient_df.at[pseudo_id, 'age'] < i[1]:
            age_group = age_groups[i]
            break

    group = lbl_discharged if patient_df.at[pseudo_id, 'group_discharged'] == 1 else lbl_died if patient_df.at[pseudo_id, 'died'] == 1 else lbl_icu
    los = patient_df.at[pseudo_id, 'hospitalisation_time'] / 24

    df.loc[pseudo_id] = [age_group, group, los]

df.sort_values('Age Group', inplace=True)
print(df.groupby(by=['Age Group', 'Group']).size().unstack(fill_value=0))

# Plot boxplots
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x="Age Group", y="LOS (days)", hue="Group", data=df, palette=[clr_discharged, clr_icu, clr_died], hue_order=hue_order)
plt.tight_layout()
save_file(fig, f"{o_path}/LOS_Age.png")


#######
# 4. LOS for the three groups & sex

df = pd.DataFrame(columns=['Sex', 'Group', 'LOS (days)'], index=patient_df.index)

# Loop through patients
for pseudo_id in patient_df.index:
    sex = 'Female' if patient_df.at[pseudo_id, 'sex'] == 'F' else 'Male'
    group = lbl_discharged if patient_df.at[pseudo_id, 'group_discharged'] == 1 else lbl_died if patient_df.at[pseudo_id, 'died'] == 1 else lbl_icu
    los = patient_df.at[pseudo_id, 'hospitalisation_time'] / 24

    df.loc[pseudo_id] = [sex, group, los]

print(df.groupby(by=['Sex', 'Group']).size().unstack(fill_value=0))

# Plot boxplots
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x="Sex", y="LOS (days)", hue="Group", data=df, palette=[clr_discharged, clr_icu, clr_died], hue_order=hue_order)
plt.tight_layout()
save_file(fig, f"{o_path}/LOS_Sex.png")
