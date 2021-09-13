import pandas as pd
import numpy as np
import re
from Toolkit import *
from dateparser import parse
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


"""
This script performs two things:
1. From raw string data in Complaints file, get the exact date of first complaints for every patients.
    This is not perfect and requires a manual check.
2. Plot multiple graphs that show length of stay in relation to symptom duration
"""


def str_to_datetime(comp, adm):
    """
    Changes strings in Pandas dataframe column to a proper DateTime object
    Assumes values of type "01-01" or "01-01-01" with both "/" and "-" as divise symbols
    """

    for i in range(len(comp)):
        if not pd.isna(comp[i]) and not isinstance(comp[i], datetime) and not comp[i][-8:] == '00:00:00':
            raw_date = comp[i].replace('/', '-')
            # Make DateTime object from date string
            if len(raw_date) <= 5:
                raw_date = raw_date + '-20'
            comp[i] = datetime.strptime(raw_date, '%d-%m-%y')
            # Reset year to 2021 if complaints day is 31 or more days before admission
            if (adm[i] - comp[i]).days > 31 and adm[i].year == 2021:
                print(comp[i], adm[i])
                comp[i] = comp[i].replace(year=2021)
    return comp


# Parameters
fontsize = 16
plt.rcParams.update({'font.size': fontsize})
o_df_path = "Processed_Data/first_complaint_dates.csv"

# Read complaints and patients stats data files
complaints_df = exclude("covid-studie-rachel-klachten.csv", "Stats/Patient_Statistics_Haga.csv", symptom_onset=False, treatment_limit=False, dis_other_hospital=False)
time = 'klachten_form_entries_start_date'
complaints_df[time] = pd.to_datetime(complaints_df[time])

patient_df = exclude("Stats/Patient_Statistics_Haga.csv", "Stats/Patient_Statistics_Haga.csv", symptom_onset=False, treatment_limit=False, dis_other_hospital=False)
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
patient_df.set_index('pseudo_id', inplace=True)
patient_df.sort_values(['hospitalisation_time'], ascending=True, inplace=True)

if not path.isfile(o_df_path):
    # Make output df
    o_df = pd.DataFrame(columns=['complaints_date', 'admission_date', 'discharge_date', 'logged_date', 'raw_string'], index=[i for i in complaints_df['pseudo_id'].unique() if i in patient_df.index])
else:
    o_df = pd.read_csv(o_df_path)
    o_df.set_index('Unnamed: 0', inplace=True)

# List of all regex strings to filter out first day of complaints
regexs = [r"dag\d\d?vanklachten",  # 0-0 - relative symptoms day, keep this on top
          r"klachten((sinds)|(dag)|(datum)).{0,8}?\d\d?[/-]\d\d?([/-]\d\d(\d\d)?)?(?!dagen)",  # 1 - absolute symtoms day
          r"((klachte[nr])|(ziektedag)).{0,8}?\d\d?[/-]\d\d?([/-]\d\d(\d\d)?)?(?!(\d)?positievetest)",  # 2-1 - absolute symptoms day
          r"klachten((dag)|(sinds)).{0,8}?\d\d?",  # 3 - relative symtoms day
          r"is.{0,6}?\d\d?dag(en)?ziek",  # 4-2 - relative symtoms day
          r"((sinds)|(klachten)).{0,8}?\d\d?",  # 5-3 - relative symtoms day
          r"\d\d?[/-]\d\d?(\d\d(\d\d)?)?.{0,5}?respiratoireklachten",  # 6-4 - absolute symtoms day
          r"sinds((vorige)|(afgelopen))?(week)?[a-z]{4,6}dag(?!(covidpositief)|(positief))"]  # 7-5 - e.g. "afgelopen zaterdag klachten"

# Check with every regular expression for matches
for regex_nr, regex in enumerate(regexs):
    for row in complaints_df.index:
        pseudo_id = complaints_df.at[row, 'pseudo_id']

        if pseudo_id in patient_df.index and pd.isna(o_df.loc[pseudo_id, 'complaints_date']):
            o_df.loc[pseudo_id, 'admission_date'] = patient_df.at[pseudo_id, 'admitted_date']
            o_df.loc[pseudo_id, 'discharge_date'] = patient_df.at[pseudo_id, 'discharge_date']

            complaint = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '').replace(' ', '')

            find = re.search(regex, complaint)
            if find:
                # Search for date in returned string
                if regex_nr in [1, 2, 6]:  # absolute day
                    find_date = re.search(r"\d\d?[/-]\d\d?", find.group(0))
                    if find_date:
                        o_df.loc[pseudo_id, 'raw_string'] = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '')
                        o_df.loc[pseudo_id, 'complaints_date'] = find_date.group(0)
                        o_df.loc[pseudo_id, 'logged_date'] = patient_df.at[pseudo_id, 'admitted_date'] + timedelta(hours=round(
                            (complaints_df.loc[row, time]-patient_df.at[pseudo_id, 'admitted_date']).total_seconds() / 3600))
                    else:
                        print(f"WARNING: FOUND 1ST STRING BUT NOT SECOND STRING:\n\t{find.group(0)}")
                elif regex_nr in [0, 3, 4]:  # relative day
                    find_date = re.search(r"\+?\d\d?", find.group(0))
                    if find_date:
                        delta_days = int(find_date.group(0)[1:]) if find_date.group(0).find('+') != -1 else int(find_date.group(0))
                        admission_date = patient_df.at[pseudo_id, 'admitted_date']
                        o_df.loc[pseudo_id, 'raw_string'] = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '')
                        # Subtract found delta days (e.g. +5) from record time and add the hours offset from admission
                        newdatetime = admission_date + timedelta(days=-delta_days, hours=round((complaints_df.loc[row, time]-admission_date).total_seconds() / 3600))
                        o_df.loc[pseudo_id, 'complaints_date'] = datetime(newdatetime.year, newdatetime.month, newdatetime.day)
                        o_df.loc[pseudo_id, 'logged_date'] = patient_df.at[pseudo_id, 'admitted_date'] + \
                                                             timedelta(hours=round((complaints_df.loc[row, time]-admission_date).total_seconds() / 3600))
                    else:
                        print(f"WARNING: FOUND 1ST STRING BUT NOT SECOND STRING:\n\t{find.group(0)}")
                elif regex_nr == 5:  # relative day OR absolute day
                    find_date = re.search(r"((sinds)|(klachten)).{0,8}?\d\d?[/-]\d\d?([/-]\d\d(\d\d)?)?(?!dagen)", complaint)
                    if find_date:  # Absolute day
                        find_date = re.search(r"\d\d?[/-]\d\d?", find_date.group(0))
                        o_df.loc[pseudo_id, 'raw_string'] = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '')
                        o_df.loc[pseudo_id, 'complaints_date'] = find_date.group(0)
                        o_df.loc[pseudo_id, 'logged_date'] = patient_df.at[pseudo_id, 'admitted_date'] + timedelta(hours=round(
                            (complaints_df.loc[row, time]-patient_df.at[pseudo_id, 'admitted_date']).total_seconds() / 3600))
                    else:
                        find_date = re.search(r"\+?\d\d?", find.group(0))
                        if find_date:  # Relative day
                            delta_days = int(find_date.group(0)[1:]) if find_date.group(0).find('+') != -1 else int(find_date.group(0))
                            admission_date = patient_df.at[pseudo_id, 'admitted_date']
                            o_df.loc[pseudo_id, 'raw_string'] = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '')
                            # Subtract found delta days (e.g. +5) from record time and add the hours offset from admission
                            newdatetime = admission_date + timedelta(days=-delta_days, hours=round((complaints_df.loc[row, time]-admission_date).total_seconds() / 3600))
                            o_df.loc[pseudo_id, 'complaints_date'] = datetime(newdatetime.year, newdatetime.month, newdatetime.day)
                            o_df.loc[pseudo_id, 'logged_date'] = patient_df.at[pseudo_id, 'admitted_date'] + \
                                                                 timedelta(hours=round((complaints_df.loc[row, time]-admission_date).total_seconds() / 3600))
                    # Ultimate "Else" statement that only is executed if no break line is executed
                    print(f"WARNING: FOUND 1ST STRING BUT NOT SECOND STRING:\n\t{find.group(0)}")
                elif regex_nr == 7:  # weekday (e.g. "afgelopen zaterdag")
                    find_date = re.search(r"((maan)|(dins)|(maan)|(woens)|(donder)|(vrij)|(zater)|(zon))dag", find.group(0))  # Find weekday
                    is_lastweek = re.search(r"((vorige)|(afgelopen))week", find.group(0))
                    if find_date:
                        if parse(find_date.group(0)):
                            admission_date = patient_df.at[pseudo_id, 'admitted_date']
                            o_df.loc[pseudo_id, 'raw_string'] = complaints_df.loc[row, 'klachten_form_entries_value_text'].lower().replace('\n', '')
                            logged_date = patient_df.at[pseudo_id, 'admitted_date'] + \
                                          timedelta(hours=round((complaints_df.loc[row, time]-admission_date).total_seconds() / 3600))
                            weekday = parse(find_date.group(0)).isoweekday()
                            # set is_lastweek to True in cases like 'vrijdag eerste symptomen', when logged on an earlier week day
                            if not is_lastweek:
                                is_lastweek = True if weekday >= logged_date.isoweekday() else False
                            # Set symptoms day based on the logged date, where the weekday and week number (optional) are changed
                            symptoms_day = f"{logged_date.year}-{logged_date.isocalendar()[1] - (1 if is_lastweek else 0)}-{weekday}"
                            symptoms_day = datetime.strptime(symptoms_day, "%G-%V-%u")
                            o_df.loc[pseudo_id, 'complaints_date'] = symptoms_day
                            o_df.loc[pseudo_id, 'logged_date'] = logged_date
                    else:
                        print(f"WARNING: FOUND 1ST STRING BUT NOT SECOND STRING:\n\t{find.group(0)}")
    print(f"Regex nr {regex_nr}, {len(o_df[pd.isna(o_df['complaints_date'])].index)} patients left")

o_df.sort_index(inplace=True)
# Set the manually checked dates
patient_dates = pd.read_csv('klachtendag_check.csv', sep=';')
for row in patient_dates.index:
    if patient_dates.loc[row, 'pseudo_id'] in o_df.index and patient_dates.loc[row, 'pseudo_id'] in patient_df.index:
        if patient_dates.loc[row, 'date'] not in ['?', 'check']:
            if not pd.isna(patient_dates.loc[row, 'date']):
                o_df.loc[patient_dates.loc[row, 'pseudo_id'], 'complaints_date'] = datetime.strptime(patient_dates.loc[row, 'date'], '%d-%m-%Y')
            else:
                o_df.loc[patient_dates.loc[row, 'pseudo_id'], 'complaints_date'] = np.nan
            o_df.loc[patient_dates.loc[row, 'pseudo_id'], 'logged_date'] = np.nan
            o_df.loc[patient_dates.loc[row, 'pseudo_id'], 'raw_string'] = np.nan

# Change columns with string dates to datetime objects
o_df['admission_date'] = pd.to_datetime(o_df['admission_date'])
o_df['discharge_date'] = pd.to_datetime(o_df['discharge_date'])
o_df['complaints_date'] = str_to_datetime(o_df['complaints_date'], o_df['admission_date'])

# Add dates to patient_df (load the df again so no patients will be excluded in the saved file)
temp_patient_df = pd.read_csv('Stats/Patient_Statistics_Haga.csv')
temp_patient_df.set_index('pseudo_id', inplace=True)
for pseudo_id in o_df.index:
    if pseudo_id in temp_patient_df.index:
        temp_patient_df.loc[pseudo_id, 'symptoms_date'] = o_df.at[pseudo_id, 'complaints_date']
# Save output dataframe as CSV
temp_patient_df.to_csv('Stats/Patient_Statistics_Haga.csv')
# Remove file from memory
temp_patient_df = None

# Save output dataframe as CSV
o_df.to_csv(o_df_path)
print("Patients:", len(o_df.index))
print(f"Patients without a symptoms day: {len(o_df[pd.isna(o_df['complaints_date'])].index)}")

# Join two dfs about wrong/missing data: one from first_complaints_dates with patients without a symptoms date / a symptoms date that's at or after admission,
# and one df with all patients that are in patient_df but not in the processed complaints file
temp_df = o_df[(~pd.isna('complaints_date')) & (pd.to_datetime(o_df['complaints_date']) >= o_df['admission_date'])]
missings_df = o_df[pd.isna(o_df['complaints_date'])]
missings_df = pd.concat([missings_df, temp_df, pd.DataFrame(index=patient_df[patient_df.index.isin(complaints_df['pseudo_id']) == False].index)])
missings_df.to_csv("Processed_Data/missing_or_wrong_symptoms.csv")

"""
Plot graphs
"""

# Filter out only relevant patients for graphs
patient_df = exclude("Stats/Patient_Statistics_Haga.csv", "Stats/Patient_Statistics_Haga.csv")
patient_df['admitted_date'] = pd.to_datetime(patient_df['admitted_date'])
patient_df.set_index('pseudo_id', inplace=True)
patient_df.sort_values(['hospitalisation_time'], ascending=True, inplace=True)

# 1. Plot start of symptoms + hospitalisation for the whole of 2020
# Filter out patients with unknown symptoms day, sort by symptoms day
o_df = o_df[o_df.index.isin(patient_df.index)]
o_df['complaints_date'] = pd.to_datetime(o_df['complaints_date'])
o_df.sort_values(['complaints_date'], ascending=False, inplace=True, ignore_index=False)
# Make range of datetimes for every day in 2020
days = pd.date_range(datetime(2020, 1, 1), datetime(2021, 3, 1), freq='1D')

fig, ax = plt.subplots(figsize=(16, 12))
row = 1
labels = {'Deceased': 0, 'Discharged after ICU': 0, 'Discharged (no ICU)': 0}
# For every patient, plot two lines
for pseudo_id in o_df.index:
    print(pseudo_id)
    print(o_df.loc[pseudo_id, 'admission_date'].year)
    print(o_df.loc[pseudo_id, 'admission_date'].month)
    print(o_df.loc[pseudo_id, 'admission_date'].day)

    o_df.loc[pseudo_id, 'complaints_date'] = pd.to_datetime(o_df.loc[pseudo_id, 'complaints_date'])
    symptoms_day = datetime(o_df.loc[pseudo_id, 'complaints_date'].year, o_df.loc[pseudo_id, 'complaints_date'].month,
                            o_df.loc[pseudo_id, 'complaints_date'].day).timetuple().tm_yday + (366 if o_df.loc[pseudo_id, 'complaints_date'].year == 2021 else 0)
    admission_day = datetime(o_df.loc[pseudo_id, 'admission_date'].year, o_df.loc[pseudo_id, 'admission_date'].month,
                             o_df.loc[pseudo_id, 'admission_date'].day).timetuple().tm_yday + (366 if o_df.loc[pseudo_id, 'admission_date'].year == 2021 else 0)
    discharge_day = datetime(o_df.loc[pseudo_id, 'discharge_date'].year, o_df.loc[pseudo_id, 'discharge_date'].month,
                             o_df.loc[pseudo_id, 'discharge_date'].day).timetuple().tm_yday + (366 if o_df.loc[pseudo_id, 'discharge_date'].year == 2021 else 0)

    # if admission_day >= symptoms_day:
    if patient_df.loc[pseudo_id, 'died'] == 1:
        clr, label = colour_scheme('died')
    elif patient_df.loc[pseudo_id, 'died'] == 0 and patient_df.loc[pseudo_id, 'group_ICU'] == 1:
        clr, label = colour_scheme('ICU')
    else:
        clr, label = colour_scheme('discharged')

    ax.plot([symptoms_day, admission_day], [row, row], ls=':', c=clr, label=label + ' (symptoms)' if labels[label] == 0 else None)
    ax.plot([admission_day, discharge_day], [row, row], c=clr, label=label + ' (hospitalised)' if labels[label] == 0 else None)
    labels[label] = 1
    row += 2

ax.set_xticks([i for i in range(0, len(days), 21)])
ax.set_xticklabels([days[i].strftime("%d/%m/%Y") for i in range(0, len(days), 21)], rotation=45, ha='right')
ax.set_yticks([])
ax.set_ylabel("Patients")
ax.set_title("Symptoms and hospitalisation duration for COVID-19 patients in the Hagaziekenhuis")
ax.legend()
save_file(plt, "Stats/Symptoms_and_Hospitalisation_Duration_Haga.png")

# 2. Plot relative hospitalisation times for all patients
# For every patient, plot 1 line
fig, ax = plt.subplots(figsize=(16, 10))
row = 1
labels = {'Deceased': 0, 'Discharged after ICU': 0, 'Discharged (no ICU)': 0}
for pseudo_id in [i for i in patient_df.index if i in o_df.index]:
    # if admission_day >= symptoms_day:
    # get admission/discharge day relative to start of symptoms
    rel_admission_day = (o_df.loc[pseudo_id, 'admission_date'] - o_df.loc[pseudo_id, 'complaints_date']).days
    rel_discharge_day = (o_df.loc[pseudo_id, 'discharge_date'] - o_df.loc[pseudo_id, 'complaints_date']).days

    if patient_df.loc[pseudo_id, 'died'] == 1:
        clr, label = colour_scheme('died')
    elif patient_df.loc[pseudo_id, 'group_ICU'] == 1 and patient_df.at[pseudo_id, 'died'] == 0:
        clr, label = colour_scheme('ICU')
    elif patient_df.loc[pseudo_id, 'group_discharged'] == 1:
        clr, label = colour_scheme('discharged')
    ax.plot([rel_admission_day, rel_discharge_day], [row, row], c=clr, label=label if labels[label] == 0 else None)

    labels[label] = 1
    row += 2

ax.set_xlabel("Days after first symptoms")
ax.set_yticks([])
ax.set_ylabel("Patients")
ax.set_title("Hospitalisation duration relative to symptom onset per COVID-19\npatient in the Hagaziekenhuis")
ax.legend()
save_file(plt, "Stats/Hospitalisation_Relative_to_Symptoms_Haga.png")

"""
id
opname
symptoomdag
time
value
"""
print(f"{len(patient_df[patient_df.index.isin(complaints_df['pseudo_id']) == False].index)} patients are "
      "not present in the original complaints file.")
