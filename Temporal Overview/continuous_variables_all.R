setwd("/exports/reum/tverheijen/Combined")
set.seed(42)

library(ggplot2)
library(dplyr)
library(devtools)
library(insight)
library(sjmisc)
library(plotly)
library(reshape2)
library(RColorBrewer)
theme_set(theme_minimal())


plot_time_pattern = function(data, varX, xlabel, fname, time_list){
  
  # Plot continous variables over time with a smoothing method over time
  # var data: data.frame
  # var varX: variable name (column name) that needs to be plotted
  # var fname: output file name, excluding the time and extension
  # var time_list: list of time steps in days to plot
  
  for (timestep in time_list){
    time_var = paste('t', timestep, 'D', sep='')
    plt = ggplot(data, aes_string(x=time_var, y=varX, group = 'outcome' , shape='outcome', colour='outcome')) +
          geom_point() + geom_line(aes(linetype=outcome), size=1) +
          stat_smooth() + scale_color_manual(values=c('#008AFE', '#AB15AC', '#FE0053') ) +
          xlab(xlabel)
    if (varX == '`ALAT (U/L)`'){
      plt = plt + ylim(0, 700)
    }
    if (varX == '`ASAT (U/L)`'){
      plt = plt + ylim(0, 1250)
    }
    if (varX == '`D-Dimer (ng/mL)`'){
      plt = plt + ylim(0, 100000)
    }
    if (varX == '`Ferritin (µg/L)`'){
      plt = plt + ylim(0, 15000)
    }
    if (varX == '`Temperature (°C)`'){
      plt = plt + ylim(25, 45)
    }
    if (varX %in% c('`Creatinine SER (µmol/L)`', '`Ferritin (µg/L)`', '`Temperature (°C)`', '`Diastolic BP (mmHg)`') == FALSE){
      print("NOT printing a legeld for variable: ")
      print(varX)
      plt = plt + theme(legend.position = "none")
    }
    
    ggsave(paste(fname, paste(24 * timestep, '.png', sep=''), sep="_"), )
    print(paste("Saved file: ", paste(fname, paste(24 * timestep, '.png', sep=''), sep="_"), sep=""))
  }
}

# Define parameters
xlabel = 'Time after admission (days)'
time_col = 'time_adm'

# Load data
patient_df = read.csv("/exports/reum/tverheijen/Combined/Patient_Statistics_All.csv", 
                      check.names=FALSE, encoding="UTF-8")
vitals_df = read.csv("/exports/reum/tverheijen/Combined/vitals_before_IC.csv", 
                     check.names=FALSE, encoding="UTF-8")
lab_df = read.csv("/exports/reum/tverheijen/Combined/lab_before_IC.csv", 
                  check.names=FALSE, encoding="UTF-8")

# Change time column unit to days 
vitals_df[time_col] = vitals_df[time_col] / 24
lab_df[time_col] = lab_df[time_col] / 24

# Define granularities
granularities = c(0.25, 1)  # days
# Add column for granularity
for (timestep in granularities){
  time_var = paste('t', timestep, 'D', sep='')
  vitals_df[time_var] = group_var(vitals_df[[time_col]], size = timestep, as.num = T) * timestep
  lab_df[time_var] = group_var(lab_df[[time_col]], size = timestep, as.num = T) * timestep
}

# 1. LAB --> Plot various variables
fpath = "/exports/reum/tverheijen/Combined/Figures/ContinuousVars/Lab/"
for (feat in names(lab_df)){
  if (feat %in% c('pseudo_id', 'time', 'time_adm', "t0.25D", "t1D", "outcome") == FALSE){
    o_path = paste0(fpath, strsplit(feat, "[.,(]")[[1]][1])
    feat_str = paste0("`", feat, "`")
    plot_time_pattern(lab_df, feat_str, fname=o_path, time_list=granularities, xlabel=xlabel)
  }
}

# 2. VITALS --> Plot various variables
fpath = "/exports/reum/tverheijen/Combined/Figures/ContinuousVars/Vitals/"
for (feat in names(vitals_df)){
  if (feat %in% c('pseudo_id', 'time', 'time_adm', "t0.25D", "t1D", "outcome") == FALSE){
    o_path = paste0(fpath, strsplit(feat, "[.,(]")[[1]][1])
    feat_str = paste0("`", feat, "`")
    plot_time_pattern(vitals_df, feat_str, fname=o_path, time_list=granularities, xlabel=xlabel)
  }
}


