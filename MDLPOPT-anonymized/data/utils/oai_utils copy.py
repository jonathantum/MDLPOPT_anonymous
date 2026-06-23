import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from typing import List, Dict, Tuple, Optional, Union

class OAIDataProcessor:
    """
    A class to handle loading, processing, and analyzing OAI (Osteoarthritis Initiative) dataset.
    """
    
    def __init__(self, base_path: str = '/home/aio/OAI_Dataset/'):
        self.base_path = base_path
        self.cont_prefixes = ["00", "01", "03", "05", "06", "07", "08", "09", "10", "11"]
        self.sq_prefixes = ["00", "01", "03", "05", "06", "08", "10"]  # Semi-quantitative scoring prefixes
        
        # Define variable groups
        self.CAT_CONT_VARS = [
            "KOOSKPL", "KOOSKPR", "KOOSYML", "KOOSYMR", "KOOSFSR",
            "WOMKPL", "WOMKPR", "WOMSTFL", "WOMSTFR", "WOMADLL", "WOMADLR",
            "BMI", "BMICAT", 
            "WEIGHT", "HEIGHT", #not available for all visits
            "AGE", # only initial value
            "PASE", "PASE1HR", "PASE2HR", "PASE3HR", "PASE4HR", "PASE5HR", "PASE6HR",# physical activity scale for elderly
            "CESD", "HSMSS",#depression scale #HSMSS mental health related to physical
            "SMOKER", "SMKAGE", "SMKNEV", "SMKAVE", "SMKNOW", "SMKAMT", "SMKSTOP",  # only only 3 visits 00, 06, 10    
            "DRNKAMT",  # alcohol, only 3 visits 00, 06, 10
            "RKALNMT", "LKALNMT", # knee alignment in degrees
            "COMORB",
            "MARITST", #marital status, only 4 visits
            'CHNFQCV', # frequency
            'GLCFQCV', # frequency
            "V00EDCV", #education level, only baseline 00
            "V00INCOME", #income level, only baseline 00
        #     "400MTIM", #400 m walk time
        #     "LPN400W", # left knee pain during walk
        #     "RPN400W", 
        #     "AALTMNS", "AAMDMNS", "AAMVMNS", "AAVMNS" #using swartz cutoffs for physical activity as it may be more approptiate for older patients
        ]

        self.BOOL_VARS = [
            'KPMED', # Either knee, used medication for pain, aching or stiffness, past 12 months
            'KPMEDCV', # Either knee, used medication for pain, aching or stiffness more than half the days of a month, past 12 months
            'NSAIDS', 
            'NSAIDRX',
            'COXIBS',
            'NARCOT',
            'TYLEN',
            'CHON',
            'GLUC',
            'MSM',
            'SAME',
            'PNMEDT',
            'KNINJ',
            'DOXYCYC',
            'KNINJ', # injections either knee
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG", "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH", "LIVDAM", "CANCER",  #comorbidity score
            "INJR12", #Right knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "INJL12", #Left knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago

        ]
        
        # Initialize data containers
        self.clinical_dfs = []
        self.mega_df = None
        self.meds_df = None
        self.outcomes_df = None

    def load_clinical_data(self) -> List[pd.DataFrame]:
        """Load all clinical data files."""
        clinical_paths = [f'AllClinical{prefix}.txt' for prefix in self.cont_prefixes]
        
        clinical_dfs = []
        for clinical_path in clinical_paths:
            full_path = os.path.join(self.base_path, 'Clinical_Dataset', clinical_path)
            df = pd.read_csv(full_path, delimiter='|')
            clinical_dfs.append(df)
        
        self.clinical_dfs = clinical_dfs
        return clinical_dfs
   
    
    def load_semiquant_data(self) -> List[pd.DataFrame]:
        """Load semi-quantitative X-ray scoring data."""
        semiquant_dfs = []
        
        for prefix in self.sq_prefixes:
            file_path = os.path.join(
                self.base_path, 
                'Knee_Xray_Data/Semi-Quant Scoring_ASCII', 
                f'kxr_sq_bu{prefix}.txt'
            )
            df = pd.read_csv(file_path, delimiter='|')
            
            # # Filter by READPRJ = 15 if column exists
            # if "READPRJ" in df.columns:
            #     df = df[df["READPRJ"] == 15]
            
            # Standardize column names
            if f"v{prefix}XRKL" in df.columns:
                df = df.rename(columns={f"v{prefix}XRKL": f"V{prefix}XRKL"})
            
            semiquant_dfs.append(df)
        
        return semiquant_dfs

    def load_outcomes_data(self) -> pd.DataFrame:
        """Load outcomes data."""
        outcomes_path = os.path.join(self.base_path, 'All_Tabular_Data/Outcomes99.txt')
        outcomes_df = pd.read_csv(outcomes_path, delimiter='|')
        
        # Select relevant columns
        outcomes_df = outcomes_df[[
            'id', "V99ERKVSPR", "V99ERKVSAF", "V99ELKVSPR", "V99ELKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
            "V99ELKPODX", #left knee, primary pre-operative diagnosis - trauma but not very insightful
            "V99ELKTPPR", #left knee, type of partial follow-up knee replacement (lateral, medial,...) - not very insightful
            "V99ELKVSRP", #left knee, OAI visit follow-up knee replacement self-reported at
            "V99ELKTLPR",  #left knee, total or partial follow-up knee replacement
            "V99ERKPODX",
            "V99ERKTPPR",
            "V99ERKVSRP",
            "V99ERKTLPR"
        ]]
        
        # Clean surgery timing columns
        surgery_cols = ["V99ERKVSPR", "V99ERKVSAF", "V99ELKVSPR", "V99ELKVSAF"] # right knee, closest OAI contact prior to follow-up knee replacement, after knee replacement, left knee, closest OAI contact prior to follow-up knee replacement, after knee replacement
        for col in surgery_cols:
            if col in outcomes_df.columns:
                outcomes_df[col] = outcomes_df[col].str.split(':').str[0]
                outcomes_df[col] = outcomes_df[col].replace('.', pd.NA)
        
        self.outcomes_df = outcomes_df
        return outcomes_df
    def load_enrollees_data(self) -> pd.DataFrame:
        """Load enrollees data."""
        enrollees_path = os.path.join(self.base_path, 'All_Tabular_Data/Enrollees.txt')
        enrollees_df = pd.read_csv(enrollees_path, delimiter='|')
        
        # Select relevant columns
        enrollees_df = enrollees_df[[
            'ID', 
            "P02RACE", # P02RACE, only baseline 00
            "P02SEX", # only initial value P02SEX
            "V00COHORT"


        ]]
    
        self.enrollees_df = enrollees_df
        return enrollees_df

    def filter_dataframes_by_variables(self, dfs: List[pd.DataFrame], 
                                     variables: List[str], 
                                     prefixes: List[str]) -> List[pd.DataFrame]:
        """Filter dataframes to include only specified variables with dynamic prefixes."""
        filtered_dfs = []
        
        for df, prefix in zip(dfs, prefixes):
            columns_to_select = ["ID"]
            
            for var in variables:
                column_name = f"V{prefix}{var}"
                
                if column_name in df.columns:
                    columns_to_select.append(column_name)
                else:
                    print(f"Warning: Column '{column_name}' not found. Adding NaN column.")
                    df[column_name] = np.nan
                    columns_to_select.append(column_name)

            filtered_dfs.append(df[columns_to_select])
        
        return filtered_dfs

    def clean_semiquant_data(self, semiquant_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Clean and process semi-quantitative data."""
        filtered_sq_dfs = []
        def custom_dedup(group):
            readprj_col = "READPRJ" if "READPRJ" in group.columns else "readprj"
            "keep project 15 readings if available otherwise keep the first non-15 reading"
            prj15 = group[group[readprj_col] == 15]
            return prj15.iloc[[0]] if not prj15.empty else group.iloc[[0]]

        for df, prefix in zip(semiquant_dfs, self.sq_prefixes):
            # remove duplicates based on READPRJ
            df_nodups = df.groupby(["ID", "SIDE"], group_keys=False).apply(custom_dedup, include_groups=True)

            # Select relevant columns
            kl_col = f"V{prefix}XRKL"
            df_filtered = df_nodups[["ID", "SIDE", kl_col]].copy()
            

            
            filtered_sq_dfs.append(df_filtered)
        
        return filtered_sq_dfs

    def pivot_semiquant_data(self, sq_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Convert semi-quantitative data from long to wide format."""
        sq_pivoted_dfs = []
        
        for df, prefix in zip(sq_dfs, self.sq_prefixes):
            kl_col = f"V{prefix}XRKL"
            
            # Pivot the DataFrame
            pivoted_df = df.pivot(index='ID', columns='SIDE', values=kl_col)
            
            # Rename columns
            pivoted_df.columns = [f'V{prefix}R{kl_col[3:]}', f'V{prefix}L{kl_col[3:]}']
            
            # Reset index
            pivoted_df.reset_index(inplace=True)
            sq_pivoted_dfs.append(pivoted_df)
        
        return sq_pivoted_dfs

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names by replacing visit numbers with 'XX'."""
        df_copy = df.copy()
        df_copy.columns = [re.sub(r'V\d{2}', 'VXX', col) for col in df_copy.columns]
        return df_copy

    def concatenate_with_visit(self, dfs: List[pd.DataFrame], 
                             prefixes: List[str]) -> pd.DataFrame:
        """Concatenate dataframes with visit information."""
        standardized_dfs = [self.standardize_columns(df) for df in dfs]
        
        concatenated_df = pd.concat([
            df.assign(Visit=prefixes[i]) 
            for i, df in enumerate(standardized_dfs)
        ], ignore_index=True)
        
        return concatenated_df

    def convert_medication_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert medication values to boolean."""
        def convert_values(value):
            true_values = {'1', '1: Yes', 1.0}
            false_values = {'0', '0: No', 0.0}
            
            if value in true_values:
                return True
            elif value in false_values:
                return False
            else:
                return np.nan

        # Columns to exclude from conversion
        exclude_columns = ['ID', 'Visit']
        
        # Apply conversion only to non-excluded columns
        df_copy = df.copy()
        mask = ~df_copy.columns.isin(exclude_columns)
        df_copy.loc[:, mask] = df_copy.loc[:, mask].map(convert_values)
        
        return df_copy

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data to numeric, handling missing values."""
        df_clean = df.copy()
        
        # Replace various missing value representations with pd.NA
        df_clean = df_clean.map(lambda x: pd.NA if x in [' ', 'NaN', 'NULL', '.'] else x)
        
        # Convert to numeric where possible
        for col in df_clean.columns:
            if col not in ['ID', 'Visit']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean

    def create_mega_dataframe(self) -> pd.DataFrame:
        """Create the main combined dataframe."""
        # Load and process clinical data
        clinical_dfs = self.load_clinical_data()
        filtered_clinical_dfs = self.filter_dataframes_by_variables(
            clinical_dfs, self.CAT_CONT_VARS, self.cont_prefixes
        )
        
        # Fix column name inconsistency
        for i, prefix in enumerate(self.cont_prefixes):
            oldname = f"V{prefix}400MTIM"
            newname = f"V{prefix}MTIM400"
            if oldname in filtered_clinical_dfs[i].columns:
                filtered_clinical_dfs[i] = filtered_clinical_dfs[i].rename(
                    columns={oldname: newname}
                )
        
        # Load and process semi-quantitative data
        semiquant_dfs = self.load_semiquant_data()
        cleaned_sq_dfs = self.clean_semiquant_data(semiquant_dfs)
        pivoted_sq_dfs = self.pivot_semiquant_data(cleaned_sq_dfs)
        
        # Concatenate data
        concatenated_clinical = self.concatenate_with_visit(
            filtered_clinical_dfs, self.cont_prefixes
        )
        concatenated_sq = self.concatenate_with_visit(
            pivoted_sq_dfs, self.sq_prefixes
        )
        
        # Merge clinical and semi-quantitative data
        mega_df = pd.merge(
            concatenated_clinical, concatenated_sq, 
            on=['ID', 'Visit'], how='outer'
        )
        
        # Clean specific mappings
        pn400_mapping = {
            '.: Missing Form/Incomplete Workbook': np.nan, 
            "1: Yes": True, 
            1.0: True
        }
        
        if 'VXXLPN400W' in mega_df.columns:
            mega_df['VXXLPN400W'] = mega_df['VXXLPN400W'].map(pn400_mapping)
        if 'VXXRPN400W' in mega_df.columns:
            mega_df['VXXRPN400W'] = mega_df['VXXRPN400W'].map(pn400_mapping)
        
        self.mega_df = mega_df
        return mega_df

    def create_medication_dataframe(self) -> pd.DataFrame:
        """Create medication dataframe."""
        if not self.clinical_dfs:
            self.load_clinical_data()
        
        filtered_meds_dfs = self.filter_dataframes_by_variables(
            self.clinical_dfs, self.BOOL_VARS, self.cont_prefixes
        )
        
        concatenated_meds = self.concatenate_with_visit(
            filtered_meds_dfs, self.cont_prefixes
        )
        
        # Convert medication values to boolean
        meds_df = self.convert_medication_values(concatenated_meds)
        
        self.meds_df = meds_df
        return meds_df

    def get_cohorts_by_kl_grade(self, visit: str = '00', 
                               use_worst: bool = True) -> Dict[int, set]:
        """Get patient cohorts grouped by KL grade."""
        if self.mega_df is None:
            self.create_mega_dataframe()
        
        filtered_df = self.mega_df[self.mega_df['Visit'] == visit].copy()
        
        if use_worst:
            filtered_df['worstKL'] = filtered_df[['VXXLXRKL', 'VXXRXRKL']].max(axis=1)
            cohorts = {}
            for i in range(5):
                cohorts[i] = set(filtered_df[filtered_df["worstKL"] == i]['ID'].tolist())
        else:
            # Separate left and right cohorts
            left_cohorts = {}
            right_cohorts = {}
            for i in range(5):
                left_condition = (filtered_df['VXXLXRKL'] == i)
                right_condition = (filtered_df['VXXRXRKL'] == i)
                left_cohorts[i] = set(filtered_df[left_condition]['ID'].tolist())
                right_cohorts[i] = set(filtered_df[right_condition]['ID'].tolist())
            return left_cohorts, right_cohorts
        
        return cohorts

    def get_medication_cohorts(self, visit: str = '00') -> Dict[str, set]:
        """Get patient cohorts grouped by medication usage."""
        if self.meds_df is None:
            self.create_medication_dataframe()
        
        visit_data = self.meds_df[self.meds_df['Visit'] == visit]
        
        # NSAID users
        nsaids_condition = (visit_data['VXXNSAIDS'] == True)
        nsaidrx_condition = (visit_data['VXXNSAIDRX'] == True)
        coxibs_condition = (visit_data['VXXCOXIBS'] == True)
        
        nsaids_ids = set(visit_data[nsaids_condition]['ID'].tolist())
        nsaidrx_ids = set(visit_data[nsaidrx_condition]['ID'].tolist())
        coxibs_ids = set(visit_data[coxibs_condition]['ID'].tolist())
        nsaids_combined = nsaids_ids | nsaidrx_ids | coxibs_ids
        
        # Other medication types
        narcot_condition = (visit_data['VXXNARCOT'] == True)
        narcot_ids = set(visit_data[narcot_condition]['ID'].tolist())
        
        tylen_condition = (visit_data['VXXTYLEN'] == True)
        tylen_ids = set(visit_data[tylen_condition]['ID'].tolist())
        
        pnmedt_condition = (visit_data['VXXPNMEDT'] == True)
        kninj_condition = (visit_data['VXXKNINJ'] == True)
        general_ids = set(visit_data[pnmedt_condition]['ID'].tolist()) | \
                     set(visit_data[kninj_condition]['ID'].tolist())
        
        # Get all IDs and find those not using any medication
        all_ids = set(self.mega_df['ID'].tolist())
        medicated_ids = nsaids_combined | narcot_ids | tylen_ids | general_ids
        no_med_ids = all_ids - medicated_ids
        
        return {
            "nsaids": nsaids_combined,
            "narcotics": narcot_ids,
            "non_nsaids": tylen_ids,
            "general": general_ids,
            "none": no_med_ids
        }

    def save_processed_data(self, output_dir: str = "./"):
        """Save processed dataframes to CSV files."""
        if self.mega_df is not None:
            self.mega_df.to_csv(os.path.join(output_dir, "CL_OAI_variable_df.csv"), index=False)
        
        if self.meds_df is not None:
            self.meds_df.to_csv(os.path.join(output_dir, "CL_meds_df.csv"), index=False)
        
        if self.outcomes_df is not None:
            self.outcomes_df.to_csv(os.path.join(output_dir, "CL_outcomes.csv"), index=False)
            
        if self.enrollees_df is not None:
            self.enrollees_df.to_csv(os.path.join(output_dir, "CL_enrollees.csv"), index=False)


class OAIVisualizer:
    """
    A class to create visualizations for OAI data analysis.
    """
    
    def __init__(self, processor: OAIDataProcessor):
        self.processor = processor
        
    def plot_medication_by_kl_grade(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot relationship between medication usage and KL grade."""
        cohorts = self.processor.get_cohorts_by_kl_grade()
        medication_cohorts = self.processor.get_medication_cohorts()
        
        data = []
        for med, med_patients in medication_cohorts.items():
            for dis, dis_patients in cohorts.items():
                overlap = len(med_patients & dis_patients)
                total_disease_patients = len(dis_patients)
                normalized_value = overlap / total_disease_patients if total_disease_patients > 0 else 0
                
                data.append({
                    "Medication": med, 
                    "KL_Grade": dis, 
                    "Proportion": normalized_value,
                    "Count": overlap
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=df, x="KL_Grade", y="Proportion", hue="Medication", palette="Set2")
        plt.title("Medication Usage by KL Grade")
        plt.xlabel("KL Grade")
        plt.ylabel("Proportion of Patients Using Medication")
        plt.legend(title="Medication Type")
        plt.show()

    def plot_longitudinal_symptoms(self, cohorts: Dict, variable: str, 
                                 title: str, ylabel: str, 
                                 labels: List[str] = None,
                                 figsize: Tuple[int, int] = (12, 6)):
        """Plot longitudinal symptoms for different patient cohorts."""
        if labels is None:
            labels = [f"Group {i}" for i in range(len(cohorts))]
        
        plt.figure(figsize=figsize)
        
        for i, (cohort_key, cohort_ids) in enumerate(cohorts.items()):
            if isinstance(cohort_ids, set):
                cohort_ids = list(cohort_ids)
            
            relevant = self.processor.mega_df[
                self.processor.mega_df['ID'].isin(cohort_ids)
            ].copy()
            
            # Clean data
            relevant = self.processor.clean_numeric_data(relevant)
            
            # Group by visit and calculate mean/std
            grouped = relevant.groupby('Visit')[variable].agg(['mean', 'std'])
            
            label = labels[i] if i < len(labels) else f"Group {i}"
            plt.errorbar(
                grouped.index, grouped['mean'], yerr=grouped['std'],
                fmt='-o', capsize=5, label=label
            )
        
        plt.title(title)
        plt.xlabel('Visit')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_stacked_bars(self, list_of_lists: List[List], 
                         bar_labels: List[str], 
                         title: str = 'Stacked Bar Chart',
                         figsize: Tuple[int, int] = (10, 6)):
        """Plot stacked bars representing distribution of values in lists."""
        num_bars = len(list_of_lists)
        bar_width = 0.8
        bar_positions = np.arange(num_bars)
        
        colors = plt.cm.get_cmap('tab10', 10)
        
        plt.figure(figsize=figsize)
        
        handles = []
        labels = []
        
        for i, lst in enumerate(list_of_lists):
            # Remove NaN values
            lst_clean = [x for x in lst if pd.notna(x)]
            if not lst_clean:
                continue
                
            unique_vals, counts = np.unique(lst_clean, return_counts=True)
            
            bottom = 0
            for j, count in enumerate(counts):
                bar = plt.bar(bar_positions[i], count, bottom=bottom, 
                            width=bar_width, color=colors(j))
                bottom += count
                
                if i == 0:  # Add legend only once
                    handles.append(bar)
                    labels.append(f"{unique_vals[j]}")
        
        plt.title(title)
        plt.xticks(bar_positions, bar_labels)
        plt.legend(handles=handles, labels=labels, 
                  loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def plot_accelerometer_activity(self, visit: str = '06', 
                                  figsize: Tuple[int, int] = (12, 8)):
        """Plot accelerometer activity by KL grade."""
        cohorts = self.processor.get_cohorts_by_kl_grade(visit=visit)
        
        # Filter for patients with accelerometer data
        filtered_df = self.processor.mega_df[
            (self.processor.mega_df["VXXAAMVMNS"].notna()) & 
            (self.processor.mega_df["VXXAAMDMNS"].notna()) & 
            (self.processor.mega_df['Visit'] == visit)
        ].copy()
        
        filtered_df = self.processor.clean_numeric_data(filtered_df)
        
        segment_columns = ['VXXAALTMNS', 'VXXAAMDMNS', 'VXXAAMVMNS', 'VXXAAVMNS']
        labels = ["Light Activity", "Moderate Activity", 
                 "Moderate-Vigorous Activity", "Vigorous Activity"]
        colors = ['#c9ada7', '#9a8c98', '#4a4e69', "#22223b"]
        
        plt.figure(figsize=figsize)
        for i in range(4):
            plt.subplot(2, 2, i+1)
            
            kl_grades = []
            avg_activities = []
            
            for kl_grade in range(5):
                if kl_grade in cohorts:
                    cohort_data = filtered_df[
                        filtered_df['ID'].isin(cohorts[kl_grade])
                    ]
                    if not cohort_data.empty:
                        avg = cohort_data[segment_columns[i]].mean()
                        kl_grades.append(kl_grade)
                        avg_activities.append(avg)
            
            plt.bar(kl_grades, avg_activities, color=colors[i], width=0.6)
            plt.title(labels[i])
            plt.xlabel('KL Grade')
            plt.ylabel('Average Daily Minutes')
            plt.xticks(range(5))
        
        plt.tight_layout()
        plt.show()

class OAIAnalyzer:
    """
    A class to perform various analyses on OAI data.
    """
    
    def __init__(self, processor: OAIDataProcessor):
        self.processor = processor
        
    def find_progressors(self, initial_visit: str = '00', 
                        final_visit: str = '06',
                        threshold: int = 2) -> Tuple[List, List]:
        """Find patients who progressed in KL grade."""
        # Find patients with low KL at baseline
        baseline_condition = (
            (self.processor.mega_df['VXXRXRKL'] <= threshold) & 
            (self.processor.mega_df['Visit'] == initial_visit)
        )
        healthy_baseline = self.processor.mega_df[baseline_condition]['ID'].tolist()
        
        # Find progressors
        progress_condition = (
            self.processor.mega_df['ID'].isin(healthy_baseline) & 
            (self.processor.mega_df['Visit'] == final_visit) & 
            (self.processor.mega_df['VXXRXRKL'] > threshold)
        )
        progressors = self.processor.mega_df[progress_condition]['ID'].tolist()
        
        # Find non-progressors
        normal_df = self.processor.mega_df[
            self.processor.mega_df['ID'].isin(healthy_baseline) & 
            ~self.processor.mega_df['ID'].isin(progressors)
        ]
        normal_ids = normal_df[normal_df['Visit'] == final_visit]['ID'].tolist()
        
        return progressors, normal_ids

    def create_function_cohorts(self, variable: str, 
                              initial_visit: str = '00',
                              final_visit: str = '11',
                              threshold: float = 10) -> List[List]:
        """Create cohorts based on functional changes."""
        before = self.processor.mega_df[
            self.processor.mega_df["Visit"] == initial_visit
        ][['ID', variable]]
        
        after = self.processor.mega_df[
            self.processor.mega_df["Visit"] == final_visit
        ][['ID', variable]]
        
        merged_df = before.merge(after, on='ID', suffixes=('_before', '_after'))
        merged_df['Difference'] = (
            merged_df[variable + '_after'] - merged_df[variable + '_before']
        )
        merged_df.dropna(inplace=True)
        
        decreasing = merged_df[
            merged_df['Difference'] < -1 * threshold
        ]['ID'].tolist()
        
        increasing = merged_df[
            merged_df['Difference'] > threshold
        ]['ID'].tolist()
        
        static = merged_df[
            (merged_df['Difference'] >= -1 * threshold) & 
            (merged_df['Difference'] <= threshold)
        ]['ID'].tolist()
        
        return [decreasing, increasing, static]

    def calculate_surgery_probability(self, patient_list: List, 
                                    surgery_ids: List) -> float:
        """Calculate probability of surgery for a patient list."""
        surgery_count = len(set(patient_list).intersection(surgery_ids))
        return surgery_count / len(patient_list) if patient_list else 0

    def get_surgery_cohorts(self) -> Tuple[List, List]:
        """Get lists of patients who had left/right knee surgery."""
        if self.processor.outcomes_df is None:
            self.processor.load_outcomes_data()
        
        left_surgery = self.processor.outcomes_df[
            self.processor.outcomes_df["V99ELKVSAF"].notna()
        ]['id'].tolist()
        
        right_surgery = self.processor.outcomes_df[
            self.processor.outcomes_df["V99ERKVSAF"].notna()
        ]['id'].tolist()
        
        return left_surgery, right_surgery
