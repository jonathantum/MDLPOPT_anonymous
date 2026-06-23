import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from typing import List, Dict, Tuple, Optional, Union
import unicodedata

from sympy import series

class OAIDataProcessor:
    """
    A class to handle loading, processing, and analyzing OAI (Osteoarthritis Initiative) dataset.
    """
    
    def __init__(self, base_path: str = '/home/aio/OAI_Dataset/'):
        self.base_path = base_path
        self.cont_prefixes = ["00", "01", "03", "05", "06", "07", "08", "09", "10", "11"]
        self.sq_prefixes = ["00", "01", "03", "05", "06", "08", "10"]  # Semi-quantitative scoring prefixes
        self.three_visit_prefixes = ["00", "06", "10"]  # Prefixes for variables only available at 3 visits

        self.clinical_vars = [
            "KOOSKPL", "KOOSKPR", "KOOSYML", "KOOSYMR", "KOOSFSR","KOOSFSL",
            "WOMKPL", "WOMKPR", "WOMSTFL", "WOMSTFR", "WOMADLL", "WOMADLR",
            "BMI", "BMICAT", 
            "WEIGHT", "HEIGHT", #not available for all visits
            "AGE", # only initial value
            "PASE", "PASE1HR", "PASE2HR", "PASE3HR", "PASE4HR", "PASE5HR", "PASE6HR",# physical activity scale for elderly
            "CESD", "HSMSS",#depression scale #HSMSS mental health related to physical
            "SMOKER", "SMKAGE", "SMKAVE", "SMKNOW", "SMKAMT", "SMKSTOP",  # only only 3 visits 00, 06, 10    
            "DRNKAMT",  # alcohol, only 3 visits 00, 06, 10
            "RKALNMT", "LKALNMT", # knee alignment in degrees
            "COMORB",
            "MARITST", #marital status, only 4 visits
            'CHNFQCV', # frequency
            'GLCFQCV', # frequency
            "EDCV", #education level, only baseline 00
            "INCOME", #income level, only baseline 00
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
            'DOXYCYC',
            'KNINJ', # injections either knee
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG", "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH", "LIVDAM", "CANCER",  #comorbidity score
            "INJR12", #Right knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "INJL12", #Left knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
        ]
        self.outcomes_vars = [
            'id',
             #"V99ERKVSPR",
              "V99ERKVSAF",
                #"V99ELKVSPR",
                  "V99ELKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
            # "V99ELKPODX", #left knee, primary pre-operative diagnosis - trauma but not very insightful
            # "V99ELKTPPR", #left knee, type of partial follow-up knee replacement (lateral, medial,...) - not very insightful
            # "V99ELKVSRP", #left knee, OAI visit follow-up knee replacement self-reported at
            "V99ERKTLPR",
            "V99ELKTLPR"  #left knee, total or partial follow-up knee replacement
            # "V99ERKPODX",
            # "V99ERKTPPR",
            # "V99ERKVSRP",
        ]
        self.enrollees_vars = [
            'ID', 
            "P02RACE", # P02RACE, only baseline 00
            "P02SEX", # only initial value P02SEX
            "V00COHORT",
        ]
        self.all_visits_vars = [
            "Visit",
            "KOOSKPL", "KOOSKPR", "KOOSYML", "KOOSYMR", "KOOSFSR","KOOSFSL",
            "WOMKPL", "WOMKPR", "WOMSTFL", "WOMSTFR", "WOMADLL", "WOMADLR",
            "BMI", "BMICAT", 
            "WEIGHT",  #not available for all visits
            "AGE", # only initial value
            "PASE", "PASE1HR", "PASE2HR", "PASE3HR", "PASE4HR", "PASE5HR", "PASE6HR",# physical activity scale for elderly
            "CESD", "HSMSS",#depression scale #HSMSS mental health related to physical
            "SMOKER", "SMKAGE", "SMKAVE", "SMKNOW", "SMKAMT", "SMKSTOP",  # only only 3 visits 00, 06, 10    
            "DRNKAMT",  # alcohol, only 3 visits 00, 06, 10
            "RKALNMT", "LKALNMT", # knee alignment in degrees
            "COMORB",
            "MARITST", #marital status, only 4 visits
            'CHNFQCV', # frequency
            'GLCFQCV', # frequency
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
            'DOXYCYC',
            'KNINJ', # injections either knee
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG", "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH", "LIVDAM", "CANCER",  #comorbidity score
            "INJR12", #Right knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "INJL12", #Left knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "ERKVSAF",
            "ELKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
            "ERKTLPR",
            "ELKTLPR",  #left knee, total or partial follow-up knee replacement
            "LXRKL",
            "RXRKL",

            "RACE", # P02RACE, only baseline 00
            "SEX", # only initial value P02SEX
            "COHORT",
            "HEIGHT",
            "EDCV", #education level, only baseline 00
            "INCOME", #income level, only baseline 00
        ]

        self.special_static_vars = [
            "AGE", # only initial value
            "MARITST", #marital status, only 4 visits
            "RACE", # P02RACE, only baseline 00
            "SEX", # only initial value P02SEX
            "COHORT",
            "HEIGHT",
            "EDCV", #education level, only baseline 00
            "INCOME", #income level, only baseline 00#'ID', 

        ]
        self.baseline_only_vars = [
            "AGE", # only initial value
            "MARITST", #marital status, only 4 visits
            "RACE", # P02RACE, only baseline 00
            "SEX", # only initial value P02SEX
            "COHORT",
            "HEIGHT",
            "EDCV", #education level, only baseline 00
            "INCOME", #income level, only baseline 00#'ID', 
            "KOOSKPL/KOOSKPR","KOOSYML/KOOSYMR","KOOSFSL/KOOSFSR",
            "WOMSTFL/WOMSTFR","WOMADLL/WOMADLR",
            "LKALNMT/RKALNMT","LXRKL/RXRKL"
            "CESD", "HSMSS",#depression scale #HSMSS mental health related to physical
            "COMORB",
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG", "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH", "LIVDAM", "CANCER",  #comorbidity score

            # simulation possible?
            "INJL12/INJR12",
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
            'DOXYCYC',
            'CHNFQCV', # frequency
            'GLCFQCV', # frequency
            'KNINJ', # injections either knee

        ]    
        self.simulation_vars = [
            "BMI", "BMICAT", 
            "WEIGHT",  #not available for all visits
            # "PASE", "PASE1HR", "PASE2HR", "PASE3HR", "PASE4HR", "PASE5HR", "PASE6HR",# physical activity scale for elderly
            "SMOKER", "SMKAGE", "SMKAVE", "SMKNOW", "SMKAMT", "SMKSTOP",  # only only 3 visits 00, 06, 10    
            "DRNKAMT",  # alcohol, only 3 visits 00, 06, 10
            "ELKVSAF/ERKVSAF",
            "ELKTLPR/ERKTLPR"
        ]

        self.left_vars = [
            "KOOSKPL", "KOOSYML","KOOSFSL", 
            "WOMKPL", "WOMSTFL",  "WOMADLL", 
            "LKALNMT", # knee alignment in degrees
            "INJL12", #Left knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "ELKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
            "ELKTLPR",  #left knee, total or partial follow-up knee replacement
            "LXRKL",
        ]
        self.left_right_map = {
            "KOOSKPL": "KOOSKPR",
            "KOOSYML": "KOOSYMR",
            "KOOSFSL": "KOOSFSR",
            "WOMKPL": "WOMKPR",
            "WOMSTFL": "WOMSTFR",
            "WOMADLL": "WOMADLR",
            "LKALNMT": "RKALNMT",
            "INJL12": "INJR12",
            "ELKVSAF": "ERKVSAF",
            "ELKTLPR": "ERKTLPR",
            "LXRKL": "RXRKL",
        }
        self.right_vars = [
            "KOOSKPR", "KOOSYMR", "KOOSFSR",
            "WOMKPR", "WOMSTFR",  "WOMADLR", 
            "RKALNMT", # knee alignment in degrees
            "INJR12", #Right knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "ERKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
            "ERKTLPR",
            "RXRKL"
        ]

        self.binary_vars = [
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
            'DOXYCYC',
            'KNINJ', # injections either knee
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG", "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH", "LIVDAM", "CANCER",  #comorbidity score
            "INJR12", #Right knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "INJL12", #Left knee, injured badly enough to limit ability to walk for at least two days, since last visit about 12 months ago
            "ERKVSAF",
            "ELKVSAF", #before and after TKR surgery right knee, left knee (visits not years, filter out 6 months distances)
        ]
        self.categorical_vars = [
            "BMICAT", 
            "PASE1HR", "PASE2HR", "PASE3HR", "PASE4HR", "PASE5HR", "PASE6HR",# physical activity scale for elderly
            "SMOKER",  # only only 3 visits 00, 06, 10 
            "SMKNOW",    # only only 3 visits 00, 06, 10    
            "DRNKAMT",  # alcohol, only 3 visits 00, 06, 10
            "MARITST", #marital status, only 4 visits
            'CHNFQCV', # frequency
            'GLCFQCV', # frequency
            "EDCV", #education level, only baseline 00
            "INCOME", #income level, only baseline 00
            "ERKTLPR",
            "ELKTLPR",  #left knee, total or partial follow-up knee replacement
        ]
        self.continuous_vars = [
            "KOOSKPL", "KOOSKPR", "KOOSYML", "KOOSYMR", "KOOSFSR",
            "WOMKPL", "WOMKPR", "WOMSTFL", "WOMSTFR", "WOMADLL", "WOMADLR",
            "BMI",            
            "WEIGHT", "HEIGHT", #not available for all visits
            "AGE", # only initial value
            "PASE",
            "CESD", "HSMSS",#depression scale #HSMSS mental health related to physical
            "SMKAGE",  # only only 3 visits 00, 06, 10 
            "SMKAVE",  # only only 3 visits 00, 06, 10 
            "SMKAMT",  # only only 3 visits 00, 06, 10 
            "SMKSTOP",  # only only 3 visits 00, 06, 10 
            "RKALNMT", "LKALNMT", # knee alignment in degrees
            "COMORB",
        ]
        self.CONTINUOUS_COLS = [
            "BMI", "WEIGHT", "AGE", "PASE", "CESD", "HSMSS",
            "SMKAGE", "SMKAVE", "SMKAMT", "SMKSTOP",
            "COMORB", "HEIGHT",
            "KOOSKPL/KOOSKPR", "KOOSYML/KOOSYMR",
            "KOOSFSL/KOOSFSR",
            "WOMKPL/WOMKPR", "WOMSTFL/WOMSTFR",
            "WOMADLL/WOMADLR",
            "LKALNMT/RKALNMT"
        ]

        self.ordinal_vars = [

            "LXRKL",
            "RXRKL"
        ]
        self.ORDINAL_COLS = [
            "BMICAT",
            "PASE1HR", "PASE2HR", "PASE3HR",
            "PASE4HR", "PASE5HR", "PASE6HR",
            "SMOKER", "SMKNOW",
            "DRNKAMT",
            "MARITST",
            "CHNFQCV", "GLCFQCV",
            "EDCV", "INCOME",
            "ELKTLPR/ERKTLPR",
            "LXRKL/RXRKL"
        ]
        self.BINARY_COLS = [
            "KPMED", "KPMEDCV", "NSAIDS", "NSAIDRX", "COXIBS",
            "NARCOT", "TYLEN", "CHON", "GLUC", "MSM", "SAME",
            "PNMEDT", "DOXYCYC", "KNINJ",
            "HRTAT", "BYPLEG", "STROKE", "ASTHMA", "LUNG",
            "ULCER", "DIAB", "KIDFXN", "RA", "POLYRH",
            "LIVDAM", "CANCER",
            "INJL12/INJR12",
            "ELKVSAF/ERKVSAF"
        ]
        self.NOMINAL_COLS = ["SEX", "RACE", "COHORT"]

        # Initialize data containers
        self.clinical_dfs = []
        self.clinical_df = None
        self.kxr_df = None
        self.outcomes_df = None
        self.enrollees_df = None
        self.all_visits_df = None
        self.baseline_df = None
        self.all_visits_clean_df = None
        self.all_visits_clean_simulation_df = None

        self.MISSING_TOKENS = {
        " ","", ".", "NA", "NaN", "nan",
        ".: Missing Form/Incomplete Workbook"
        }


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
        outcomes_df = outcomes_df[self.outcomes_vars]
        
        # Clean surgery timing columns
        surgery_cols = ["V99ERKVSPR", "V99ERKVSAF", "V99ELKVSPR", "V99ELKVSAF"] # right knee, closest OAI contact prior to follow-up knee replacement, after knee replacement, left knee, closest OAI contact prior to follow-up knee replacement, after knee replacement
        for col in surgery_cols:
            if col in outcomes_df.columns:
                outcomes_df[col] = outcomes_df[col].str.split(':').str[0]
                outcomes_df[col] = outcomes_df[col].replace('.', pd.NA)
                outcomes_df[col] = outcomes_df[col].replace({
                    '2': '3',
                    '4': '5'
                    })
        return outcomes_df
    
    def create_outcomes_dataframe(self) -> pd.DataFrame:
        """
        Creates an outcomes table with one row per (ID, VisitID):
        - VXXERKVSAF = 1 only at the visit where right-knee surgery happened
        - VXXERKTLPR = 'partial' or 'total' only at that same visit
        - Same for left knee (VXXELKVSAF, VXXELKTLPR)
        """

        outcomes = self.load_outcomes_data().copy()

        # Convert visit columns to numeric visit IDs
        for col in ["V99ERKVSAF", "V99ELKVSAF"]:
            if col in outcomes.columns:
                outcomes[col] = pd.to_numeric(outcomes[col], errors="coerce")

        # # Normalize surgery-type labels
        # for col in ["V99ERKTLPR", "V99ELKTLPR"]:
        #     if col in outcomes.columns:
        #         outcomes[col] = (
        #             outcomes[col]
        #             .astype(str)
        #             .str.lower()
        #             .replace({
        #                 "1": "partial",
        #                 "partial": "partial",
        #                 "2": "total",
        #                 "total": "total",
        #                 ".": pd.NA
        #             })
        #         )

        # ---------------------------------------------------------------------
        # Build output table
        # ---------------------------------------------------------------------
        result_rows = []

        for _, row in outcomes.iterrows():
            pid = row["id"]

            right_visit = row.get("V99ERKVSAF", pd.NA)
            left_visit  = row.get("V99ELKVSAF", pd.NA)

            right_type = row.get("V99ERKTLPR", pd.NA)
            left_type  = row.get("V99ELKTLPR", pd.NA)

            for prefix in self.cont_prefixes:
                visit = int(prefix)

                # RIGHT knee one-hot
                if pd.notna(right_visit) and visit == right_visit:
                    r_surg = 1
                    r_type = right_type
                else:
                    r_surg = 0
                    r_type = pd.NA

                # LEFT knee one-hot
                if pd.notna(left_visit) and visit == left_visit:
                    l_surg = 1
                    l_type = left_type
                else:
                    l_surg = 0
                    l_type = pd.NA

                result_rows.append({
                    "ID": pid,
                    "Visit": prefix,
                    "ERKVSAF": r_surg,
                    "ERKTLPR": r_type,
                    "ELKVSAF": l_surg,
                    "ELKTLPR": l_type,
                })

        out = pd.DataFrame(result_rows)

        self.outcomes_df = out
        return out

    def load_enrollees_data(self) -> pd.DataFrame:
        """Load enrollees data and expand to all visits."""
        enrollees_path = os.path.join(self.base_path, 'All_Tabular_Data/Enrollees.txt')
        enrollees_df = pd.read_csv(enrollees_path, delimiter='|')

        # Select relevant columns
        cols = [
            'ID',
            "P02RACE",  # baseline
            "P02SEX",   # baseline
            "V00COHORT" # baseline
        ]
        enrollees_df = enrollees_df[cols]

        # visits to create
        visits = self.cont_prefixes

        # Create an expanded table
        expanded_rows = []

        for _, row in enrollees_df.iterrows():
            pid = row["ID"]
            for visit in visits:
                if visit == '00':
                    # baseline values
                    new_row = {
                        "ID": pid,
                        "Visit": visit,
                        "RACE": row["P02RACE"],
                        "SEX": row["P02SEX"],
                        "COHORT": row["V00COHORT"],
                    }
                else:
                    # all other visits → same columns but NaN except ID + Visit
                    new_row = {
                        "ID": pid,
                        "Visit": visit,
                        "RACE": None,
                        "SEX": None,
                        "COHORT": None,
                    }
                expanded_rows.append(new_row)

        expanded_df = pd.DataFrame(expanded_rows)

        self.enrollees_df = expanded_df
        return expanded_df

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
        df_copy.columns = [re.sub(r'V\d{2}', '', col) for col in df_copy.columns]
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

    def create_kxr_dataframe(self) -> pd.DataFrame:
        """Create the kxr dataframe."""
        # Load and process semi-quantitative data
        semiquant_dfs = self.load_semiquant_data()
        cleaned_sq_dfs = self.clean_semiquant_data(semiquant_dfs)
        pivoted_sq_dfs = self.pivot_semiquant_data(cleaned_sq_dfs)
        
        # Concatenate data
        concatenated_sq = self.concatenate_with_visit(
            pivoted_sq_dfs, self.sq_prefixes
        )
        
        self.kxr_df = concatenated_sq
        return concatenated_sq

    def create_clinical_dataframe(self) -> pd.DataFrame:
        """Create the clinical dataframe."""
        # Load and process clinical data
        clinical_dfs = self.load_clinical_data()
        filtered_clinical_dfs = self.filter_dataframes_by_variables(
            clinical_dfs, self.clinical_vars, self.cont_prefixes
        )

        # Concatenate data
        concatenated_clinical = self.concatenate_with_visit(
            filtered_clinical_dfs, self.cont_prefixes
        )
        self.clinical_df = concatenated_clinical
        return concatenated_clinical

    def select_columns(self, df: pd.DataFrame, allowed_vars: list, df_name: str):

        # Normalize column list (string)
        df_cols = df.columns.tolist()

        # Columns we will keep
        keep_cols = ["ID", "Visit"]

        # Keep columns where allowed var appears anywhere in the column name
        # for var in allowed_vars:
        #     for col in df_cols:
        #         if var in col:
        #             keep_cols.append(col)
        for var in allowed_vars:
            pattern = rf'\b{var}\b'
            for col in df_cols:
                if re.search(pattern, col):
                    keep_cols.append(col)
        # Deduplicate (ID/Visit might repeat)
        keep_cols = list(dict.fromkeys(keep_cols))

        # --- Missing variables ---
        missing = [var for var in allowed_vars
                if not any(var in col for col in df_cols)]
        if missing:
            print(f"[Warning] {df_name}: Missing expected variables: {missing}")

        # --- Extra variables ---
        extra = [col for col in df_cols
                if not any(var in col for var in allowed_vars)
                and col not in ("ID", "Visit")]
        if extra:
            print(f"[Info] {df_name}: Removing extra variables: {extra}")

        # Filter dataframe
        return df[[col for col in keep_cols if col in df.columns]]
    def explode_left_right_knees(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One patient/visit row  ->  two knee-specific rows.
        Left/right knee variables are collapsed into unified columns.
        """

        # detect visit prefixes like V00, V03, V06
        prefixes = sorted({
            c[:3] for c in df.columns
            if c.startswith("V") and c[1:3].isdigit()
        })
        print("Prefixes",prefixes)
        prefixes=["VXX"]

        # detect all raw knee columns (with prefixes)
        knee_cols = set()
        for col in df.columns:

            if col in self.left_vars or col in self.right_vars:
                    knee_cols.add(col)

        print(f"Exploding knees: removing columns {knee_cols}")
        rows = []

        for _, row in df.iterrows():
            for side in ["L", "R"]:
                new_row = {}

                # ID
                new_row["ID"] = f"{row['ID']}{side}"

                # Visit
                if "Visit" in df.columns:
                    new_row["Visit"] = row["Visit"]

                # copy ONLY non-knee columns
                for col in df.columns:
                    if col in knee_cols or col == "ID":
                        continue
                    new_row[col] = row[col]

                # add unified knee variables
                #print(f"Processing prefix: {p}")
                for l, r in self.left_right_map.items():
                    if l not in knee_cols or r not in knee_cols:
                        continue
                    unified_col = f"{l}/{r}"
                    if side == "L":
                        new_row[unified_col] = row.get(l, pd.NA)
                    else:
                        new_row[unified_col] = row.get(r, pd.NA)

                rows.append(new_row)

        exploded_df = pd.DataFrame(rows)


        # nice ordering
        front_cols = [c for c in ["ID", "Visit"] if c in exploded_df.columns]
        remaining = [c for c in exploded_df.columns if c not in front_cols]

        return exploded_df[front_cols + remaining]

    def create_all_visits_dataframe(self, clinical_df, kxr_df, outcomes_df, enrollees_df) -> pd.DataFrame:
                # Merge clinical, semi-quantitative and outcomes data that are supposed to be available at all visits
        all_visits_df = pd.merge(
            clinical_df, kxr_df,
            on=['ID', 'Visit'], how='outer'
        )     
        all_visits_df = pd.merge(
            all_visits_df, outcomes_df,
            on=['ID', 'Visit'], how='outer'
        )
        all_visits_df = pd.merge(
            all_visits_df, self.enrollees_df,
            on=['ID', 'Visit'], how='outer'
        )
        all_visits_df.sort_values(by=['ID', 'Visit'], inplace=True)

        all_visits_df = self.select_columns(
            all_visits_df, self.all_visits_vars, "All Visits DataFrame"     
        )
        all_visits_df = self.explode_left_right_knees(all_visits_df)
        
        # all_visits_df, removed_patients = self.remove_patients_with_x_missing_entries(
        #     all_visits_df, 'VXXWOMKPL/VXXWOMKPR', max_missing=2)

        # all_visits_df = self.clean_irregular_entries(all_visits_df)

        # all_visits_df = self.replace_blanks_with_nan(all_visits_df)



        # all_visits_df = self.impute_missing_entries(
        #     all_visits_df, variable= 'VXXWOMKPL', method='interpolate')
        # all_visits_df = self.impute_missing_entries(
        #     all_visits_df, variable= 'VXXWOMKPR', method='interpolate')

        

        # self.check_single_variable_missingness(all_visits_df, 'WOMKPL/WOMKPR')
        # self.check_single_variable_missingness(all_visits_df, 'VXXWOMKPR')

        self.all_visits_df = all_visits_df
        return all_visits_df

    def normalize_missing(self, series: pd.Series) -> pd.Series:
        return series.replace(list(self.MISSING_TOKENS), np.nan)
    
    def clean_continuous(self, series: pd.Series) -> pd.Series:
        series = self.normalize_missing(series)
        return pd.to_numeric(series, errors="coerce").astype("float32")
    
    def clean_binary(self, series: pd.Series) -> pd.Series:
        series = self.normalize_missing(series)

        mapping = {
            0: 0, 1: 1,
            0.0: 0, 1.0: 1,
            "0": 0, "1": 1,
            "0: No": 0,
            "1: Yes": 1
        }

        return series.map(mapping).astype("float32")

    def clean_ordinal_from_label(self, series: pd.Series) -> pd.Series:
        series = self.normalize_missing(series)

        def extract_code(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return int(x)
            match = re.match(r"^\s*(\d+)", str(x))
            return int(match.group(1)) if match else np.nan

        return series.apply(extract_code).astype("float32")

    def clean_nominal(self, series: pd.Series):
        series = self.normalize_missing(series)

        # extract label text if "1: Male"
        def normalize_label(x):
            if pd.isna(x):
                return np.nan
            if ":" in str(x):
                return str(x).split(":", 1)[1].strip()
            return str(x)

        clean = series.apply(normalize_label)
        codes, uniques = pd.factorize(clean, sort=True)
        codes = codes.astype("float32")
        codes[codes < 0] = np.nan

        return codes, uniques
    
    def clean_all_visits(self, df: pd.DataFrame):
        df = df.copy()
        encoders = {}

        for col in self.CONTINUOUS_COLS:
            if col in df:
                df[col] = self.clean_continuous(df[col])

        for col in self.ORDINAL_COLS:
            if col in df:
                df[col] = self.clean_ordinal_from_label(df[col])

        for col in self.BINARY_COLS:
            if col in df:
                df[col] = self.clean_binary(df[col])

        for col in self.NOMINAL_COLS:
            if col in df:
                df[col], encoders[col] = self.clean_nominal(df[col])

        self.all_visits_clean_df = df
        return df #, encoders


    # def create_baseline_only_dataframe(self, clinical_df, enrollee_df) -> pd.DataFrame:
    #     # Merge clinical and enrollee data that are supposed to be available only at baseline
        
    #     baseline_df = pd.merge(
    #         clinical_df, enrollee_df,
    #         on=['ID','Visit'], how='outer'
    #     )     
    #     baseline_df = self.select_columns(
    #         baseline_df, self.baseline_only_vars, "Baseline Only DataFrame"
    #     )
    #     baseline_df = self.replace_blanks_with_nan(baseline_df)
    #     baseline_df = self.explode_left_right_knees(baseline_df)
    #     baseline_df = self.prepare_baseline_for_tensor_format(baseline_df)

    #     self.baseline_df = baseline_df
    #     return baseline_df
    #     # # Clean specific mappings
    #     # pn400_mapping = {
    #     #     '.: Missing Form/Incomplete Workbook': np.nan, 
    #     #     "1: Yes": True, 
    #     #     1.0: True
    #     # }
        
    #     # if 'VXXLPN400W' in clinical_df.columns:
    #     #     clinical_df['VXXLPN400W'] = clinical_df['VXXLPN400W'].map(pn400_mapping)
    #     # if 'VXXRPN400W' in clinical_df.columns:
    #     #     clinical_df['VXXRPN400W'] = clinical_df['VXXRPN400W'].map(pn400_mapping)

    def is_valid_value(self,x):
        if pd.isna(x):
            return False
        if isinstance(x, str) and x.strip() in ["", ".", "NA", "NaN"]:
            return False
        try:
            return float(x) != 0.0
        except Exception:
            return True

    def build_simulation_df(self, df, max_samples=None) -> pd.DataFrame:
        """
        Prepares the dataframe in full tensor format:
        - applies baseline, static, and simulation logic
        - produces an inspectable [ID, T, F] dataframe
        """

        df = df.copy()

        baseline_only_vars = set(self.baseline_only_vars)
        special_static_vars = set(self.special_static_vars)
        simulation_vars = set(self.simulation_vars)

        df = df.sort_values(["ID", "Visit"])
        df["Visit"] = df["Visit"].astype(int)
        visit_list = [int(v) for v in self.cont_prefixes]

        # tensor time index → real visit
        t_to_visit = {t: v for t, v in enumerate(visit_list)}

        # real visit → tensor time index (sometimes useful)
        visit_to_t = {v: t for t, v in t_to_visit.items()}

        max_samples = max_samples
        if max_samples is not None:
            # pick first max_samples unique IDs
            allowed_ids = df["ID"].drop_duplicates().iloc[:max_samples].tolist()
            
                # always add the special ID if it's not already included
            # special_id = "9027422L"
            # if special_id not in allowed_ids:
            #     allowed_ids.append(special_id)
            df = df[df["ID"].isin(allowed_ids)]
            

        feature_cols = [c for c in df.columns if c not in ["ID", "Visit"]]

        tensor_rows = []

        for pid, g in df.groupby("ID"):
            g = g.sort_values("Visit")

            if g[g["Visit"] == 0].empty or g[g["Visit"] == 1].empty:
                print(f"[Warning] ID {pid} missing baseline or first follow-up visit. Skipping.")
                continue

            visit_rows = {
                int(v): row
                for v, row in g.set_index("Visit").iterrows()
            }

            last_valid = {col: None for col in simulation_vars}

            # precompute first appearance for special_static_vars
            first_appearance = {}
            for col in special_static_vars:
                valid_visits = g[g[col].notna()]["Visit"]
                first_appearance[col] = (
                    int(valid_visits.iloc[0]) if not valid_visits.empty else None
                )

            for t in range(10):
                row = {"ID": pid, "Visit": t}
                visit = t_to_visit[t]
                visit_row = visit_rows.get(visit)


                for col in feature_cols:

                    # ---------------- special static vars ----------------
                    if col in special_static_vars:
                        fa = first_appearance[col]
                        if t == 0 and visit_row is not None:
                            val = visit_row[col]
                            row[col] = val if self.is_valid_value(val) else pd.NA

                        elif t == 1 and fa is not None and fa > 0:
                            row[col] = visit_rows[fa][col]

                        else:
                            row[col] = pd.NA
                    # ---------------- simulation vars ----------------
                    elif col in simulation_vars:
                        if col == "ELKVSAF/ERKVSAF" or col == "ELKTLPR/ERKTLPR":
                            # special case: surgery indicator/type
                            if visit_row is None or pd.isna(visit_row[col]):
                                row[col] = pd.NA
                            else:
                                val = visit_row[col]
                                row[col] = val
                            continue
                        # ----- baseline window -----
                        if visit in (0, 1):
                            if visit_row is None or pd.isna(visit_row[col]):
                                row[col] = pd.NA
                            else:
                                val = visit_row[col]
                                row[col] = val
                                last_valid[col] = val
                            continue

                        # ----- simulation window -----
                        if visit_row is None or pd.isna(visit_row[col]):
                            row[col] = pd.NA
                        else:
                            current = visit_row[col]
                            prev = last_valid[col]

                            if prev is None:
                                # first observed after baseline
                                row[col] = 0
                            else:
                                row[col] = (
                                    1 if current > prev else
                                -1 if current < prev else
                                    0
                                )

                            last_valid[col] = current

                    # ---------------- baseline-only vars ----------------
                    else:
                        if t == 0:
                            row[col] = visit_rows.get(0, {}).get(col, pd.NA)
                        elif t == 1:
                            row[col] = visit_rows.get(1, {}).get(col, pd.NA)
                        else:
                            row[col] = pd.NA



                tensor_rows.append(row)


        tensor_df = pd.DataFrame(tensor_rows)
        tensor_df = tensor_df.set_index(["ID", "Visit"]).sort_index()
        self.all_visits_clean_simulation_df = tensor_df
        return tensor_df


    def prepare_input_tensor_format(self, max_samples=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts the pre-built simulation_df into tensors.
        """
        tensor_df = self.all_visits_clean_simulation_df.copy()

        if max_samples is not None:
            allowed_ids = (
                tensor_df
                .index.get_level_values(0)
                .unique()
                .tolist()[:max_samples]
            )
            tensor_df = tensor_df.loc[
                tensor_df.index.get_level_values(0).isin(allowed_ids)
            ]

        mask_df = (~tensor_df.isna()).astype("int8")
        input_df = tensor_df.fillna(0)

        print("\n=== Tensor structure ===")
        print("Input tensor shape:", input_df.shape)
        print("Mask tensor shape :", mask_df.shape)
        print("N samples:", input_df.index.get_level_values(0).nunique())
        print("T steps:", input_df.index.get_level_values(1).nunique())
        print("F features:", input_df.shape[1])

        return input_df, mask_df


    def inspect_tensor_entries(self, n_samples: int = 2):
        """
        Inspect all features at each time step for a few samples.
        Prints each feature for each time step T for n_samples.
        """

        if not hasattr(self, "input_tensor_df") or self.input_tensor_df is None:
            raise ValueError("No input tensor loaded. Run build_input_tensor() first.")

        tensor_df = self.input_tensor_df
        mask_df = self.input_mask_df

        # get first n_samples IDs
        ids = tensor_df.index.get_level_values(0).unique()[:n_samples]

        for pid in ids:
            print("\n" + "=" * 80)
            print(f"ID: {pid}")

            sample = tensor_df.loc[pid]
            mask = mask_df.loc[pid]

            for t in sample.index:
                print(f"\n--- Time step T={t} ---")
                row = sample.loc[t]
                mask_row = mask.loc[t]
                for col, val in row.items():
                    if col in self.simulation_vars:
                        print(f"{col:25s}: {val:8} | mask={mask_row[col]}")

    def build_input_tensor(self, max_samples=None):
        input_df, mask_df = self.prepare_input_tensor_format(
            # self.all_visits_clean_simulation_df,
            # baseline_only_vars=self.baseline_only_vars,
            # special_static_vars=self.special_static_vars,
            # simulation_vars=self.simulation_vars,
            max_samples=max_samples
        )
        self.input_tensor_df = input_df
        self.input_mask_df = mask_df
        return input_df, mask_df
    
    def inspect_tensor_with_mask(self, n_samples: int = 2):
        if self.input_tensor_df is None or self.input_mask_df is None:
            raise ValueError("Build tensor first.")

        ids = self.input_tensor_df.index.get_level_values(0).unique()[:n_samples]

        for pid in ids:
            print("\n" + "=" * 80)
            print(f"ID: {pid}")

            x = self.input_tensor_df.loc[pid]
            m = self.input_mask_df.loc[pid]

            for t in x.index:
                print(f"\n--- T={t} ---")
                for col in x.columns:
                    if col == "ELKVSAF/ERKVSAF" or col == "ELKTLPR/ERKTLPR":
                        print(f"{col:25s}: {x.loc[t, col]:8} | mask={m.loc[t, col]}")

    def prepare_target_tensor_format(
        self,
        df: pd.DataFrame,
        target_var: str,
        cont_prefixes: list[int],
        max_samples: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds target tensor and mask tensor for a single variable.

        Output:
            target_df: MultiIndex (ID, T) → target_var
            mask_df  : MultiIndex (ID, T) → {0,1}

        T corresponds to index in cont_prefixes.
        """

        df = df.copy()

        # --------------------------------------------------
        # Sort + sample limiting (Visit-based raw data)
        # --------------------------------------------------
        if "Visit" not in df.columns:
            raise ValueError("Target dataframe must contain 'Visit' column")

        df = df.sort_values(["ID", "Visit"])

        if max_samples is not None:
            allowed_ids = (
                df["ID"]
                .drop_duplicates()
                .iloc[:max_samples]
                .values
            )
            df = df[df["ID"].isin(allowed_ids)]

        cont_prefixes = sorted(cont_prefixes)

        tensor_rows = []
        mask_rows = []
        baseline_T = 2  # visits 0 and 1

        # --------------------------------------------------
        # Build targets
        # --------------------------------------------------
        for pid, g in df.groupby("ID"):
            g = g.sort_values("Visit")

            # require baseline presence (clinical baseline)
            if g[g["Visit"] == 0].empty or g[g["Visit"] == 1].empty:
                continue

            for t, visit in enumerate(cont_prefixes):
                row = {"ID": pid, "T": t}
                mask = {"ID": pid, "T": t}

                visit_row = g[g["Visit"] == visit]

                if visit_row.empty:
                    val = pd.NA
                else:
                    val = visit_row.iloc[0][target_var]

                row[target_var] = 0.0 if pd.isna(val) else float(val)

                # mask baseline steps
                if t < baseline_T  or pd.isna(val):
                    mask[target_var] = 0
                else:
                    mask[target_var] = 1


                tensor_rows.append(row)
                mask_rows.append(mask)

        # --------------------------------------------------
        # Final tensors
        # --------------------------------------------------
        target_df = (
            pd.DataFrame(tensor_rows)
            .set_index(["ID", "T"])
            .sort_index()
        )

        target_mask_df = (
            pd.DataFrame(mask_rows)
            .set_index(["ID", "T"])
            .sort_index()
        )

        # --------------------------------------------------
        # Sanity checks
        # --------------------------------------------------
        print("\n=== Target tensor ===")
        print("Shape:", target_df.shape)
        print("Mask tensor shape:", target_mask_df.shape)
        print("N samples:", target_df.index.get_level_values(0).nunique())
        print("T steps:", target_df.index.get_level_values(1).nunique())

        print("\nObserved targets per T:")
        print(target_mask_df.groupby("T")[target_var].sum())

        return target_df, target_mask_df


    def build_target_tensor(self, max_samples=None):
        target_df, target_mask_df = self.prepare_target_tensor_format(
            self.all_visits_clean_df,
            target_var="WOMKPL/WOMKPR",
            cont_prefixes=[int(p) for p in self.cont_prefixes],
            max_samples=max_samples
        )

        self.target_tensor_df = target_df
        self.target_mask_df = target_mask_df

        return target_df, target_mask_df

    def inspect_target_tensor(self, n_samples: int = 2):
        if not hasattr(self, "target_tensor_df"):
            raise ValueError("Target tensor not built.")

        ids = self.target_tensor_df.index.get_level_values(0).unique()[:n_samples]

        for pid in ids:
            print("\n" + "=" * 80)
            print(f"ID: {pid}")
            sample = self.target_tensor_df.loc[pid]
            mask = self.target_mask_df.loc[pid]

            for t in sample.index:
                val = sample.loc[t].values[0]
                m = mask.loc[t].values[0]
                print(f"T={t}: value={val}, mask={m}")

    def align_input_and_target(self):
        common_index = self.input_tensor_df.index.intersection(
            self.target_tensor_df.index
        )

        self.input_tensor_df = self.input_tensor_df.loc[common_index]
        self.input_mask_df = self.input_mask_df.loc[common_index]

        self.target_tensor_df = self.target_tensor_df.loc[common_index]
        self.target_mask_df = self.target_mask_df.loc[common_index]

        print("\nAligned tensors:")
        print("Input:", self.input_tensor_df.shape)
        print("Target:", self.target_tensor_df.shape)

    def replace_blanks_with_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace Python None
        df = df.replace({None: np.nan})

        # Replace non-breaking space explicitly (most common invisible char)
        df = df.replace("\xa0", np.nan)

        # Replace ALL whitespace-only strings (regex)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Final fallback: robust strip-based replace
        def clean_value(x):
            if isinstance(x, str):
                if unicodedata.normalize("NFKC", x).strip() == "":
                    return np.nan
            return x

        df = df.applymap(clean_value)

        return df

    def impute_missing_entries(self, df: pd.DataFrame, variable: str, method: str = 'interpolate'):
        """
        Impute missing entries for a given variable using specified method.

        Args:
            df (pd.DataFrame): The input dataframe.
            variable (str): The variable/column name to impute.
            method (str): The imputation method. Options are 'interpolate', 'forward_fill', 'zero_fill'.
        
            """
        for col in df.columns:
            if variable in col:
                if method == 'interpolate':
                    # Ensure column is numeric (convert object -> float)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Group by ID and interpolate
                    df[col] = df.groupby('ID')[col].transform(
                        lambda group: group.interpolate(method='linear').ffill().bfill()
                    )

                #     df[col] = df.groupby('ID')[col].apply(lambda group: group.interpolate(method='linear').ffill().bfill())
                elif method == 'forward_fill':
                    df[col] = df.groupby('ID')[col].apply(lambda group: group.ffill())
                elif method == 'zero_fill':
                    df[col] = df[col].fillna(0)
                else:
                    raise ValueError(f"Unknown imputation method: {method}")
        

        return df 

    def check_single_variable_missingness(self, df: pd.DataFrame, variable: str):
        """
        For a single variable, compute how many patients have
        0,1,2,3,... missing visits.

        Returns:
            summary (Series): index = number of missing visits,
                            value = number of patients
        """

        if "ID" not in df.columns:
            raise ValueError("DataFrame must contain an 'ID' column.")
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame.")

        # True/False missing flags per visit
        missing_flags = df[variable].isna()
        missing_flags.index = df["ID"]   # index by patient

        # Count missing visits per patient
        missing_count_per_patient = missing_flags.groupby(missing_flags.index).sum()

        # Count how many patients have 0,1,2,3,... missing visits
        summary = missing_count_per_patient.value_counts().sort_index()

        print(f"\n### Missing visit summary for variable: {variable} ###")
        print(summary)

        return summary

    def remove_patients_with_x_missing_entries(self, df: pd.DataFrame, variable: str, max_missing: int):
        """
        Remove ALL visits of patients who have more than `max_missing`
        missing entries of `variable`.

        Args:
            df (pd.DataFrame): The full all_visits_df table.
            variable (str): Variable to check missingness for.
            max_missing (int): Maximum allowed number of missing visits.
                            Patients exceeding this threshold are removed.

        Returns:
            cleaned_df (pd.DataFrame): DataFrame with selected patients removed.
            removed_patients (list): List of patient IDs that were removed.
        """

        if "ID" not in df.columns:
            raise ValueError("DataFrame must contain an 'ID' column.")
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in DataFrame.")

        # Boolean missing flags per row
        missing_flags = df[variable].isna()
        missing_flags.index = df["ID"]  # index by patient ID

        # Count missing visits per patient
        missing_count_per_patient = missing_flags.groupby(missing_flags.index).sum()

        # Find patients exceeding threshold
        removed_patients = missing_count_per_patient[missing_count_per_patient > max_missing].index.tolist()

        print(f"\n### Removing patients with more than {max_missing} missing visits for {variable} ###")
        print(f"Patients removed ({len(removed_patients)}): {removed_patients}")

        # Filter them out from the full dataframe
        cleaned_df = df[~df["ID"].isin(removed_patients)].copy()

        return cleaned_df, removed_patients

    def list_unique_entries(self, df: pd.DataFrame):
        """
        List all unique entries for all columns in the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
        
        Returns:
            unique_dict (dict): A dictionary where each key is a column name
                                and each value is a sorted list of unique entries.
        """

        unique_dict = {}

        for col in df.columns:
            # Extract unique values
            uniques = df[col].dropna().unique()

            # Convert numpy types to Python types for readability
            uniques_list = sorted(uniques.tolist(), key=lambda x: (str(type(x)), x))

            unique_dict[col] = uniques_list

            print(f"\n### {col} ###")
            print(uniques_list)

        return unique_dict

    def analyze_variable_categories(self, df: pd.DataFrame):
        """
        Check that each column belongs to exactly one category (binary, categorical,
        continuous, ordinal) based on substring matching, list unique values per column,
        and report columns that do not match any category.
        """

        # Dictionary to hold results
        result = {
            "binary": {},
            "categorical": {},
            "continuous": {},
            "ordinal": {},
            "unmatched": [],
            "multi_matched": {}
        }

        # Helper dictionary for iteration
        categories = {
            "binary": self.binary_vars,
            "categorical": self.categorical_vars,
            "continuous": self.continuous_vars,
            "ordinal": self.ordinal_vars
        }

        # Check all columns
        for col in df.columns:

            matches = []

            # Check which category the column belongs to
            for cat_name, base_names in categories.items():
                for token in base_names:
                    if token in col:           # substring match
                        matches.append(cat_name)
                        break                  # stop after first match in this category

            # Handle matching cases
            if len(matches) == 0:
                result["unmatched"].append(col)
                continue

            if len(matches) > 1:
                result["multi_matched"][col] = matches
                continue

            # Exactly one match
            category = matches[0]

            # List unique values for this column
            uniques = sorted(df[col].dropna().unique().tolist(), key=lambda x: str(x))

            result[category][col] = uniques

        # ---- PRINT SUMMARY ----
        print("\n========== VARIABLE CATEGORY SUMMARY ==========")

        for cat in ["binary", "categorical", "continuous", "ordinal"]:
            print(f"\n--- {cat.upper()} VARIABLES ---")
            for col, uniques in result[cat].items():
                print(f"{col}: {uniques[:20]}" + (" ..." if len(uniques) > 20 else ""))

        if result["unmatched"]:
            print("\n⚠️ Columns with NO MATCH in any category:")
            for col in result["unmatched"]:
                print("  -", col)

        if result["multi_matched"]:
            print("\n❌ Columns matching MULTIPLE categories (fix needed):")
            for col, matches in result["multi_matched"].items():
                print(f"  - {col}: {matches}")

        print("\n===============================================\n")

        return result

    def check_irregular_entries(self, df: pd.DataFrame):
        """
        Check all columns in df and list irregular entries for each column
        based on its assigned category (binary, categorical, continuous, ordinal).

        Returns:
            irregular_dict: dict with keys = category, values = dict {column: [irregular_entries]}
        """

        # First, categorize columns using your previous function
        result = self.analyze_variable_categories(df)

        irregular_dict = {
            "binary": {},
            "categorical": {},
            "continuous": {},
            "ordinal": {}
        }

        # ---- BINARY ----
        allowed_binary = [True, False]
        for col in result["binary"]:
            values = df[col].dropna().unique()
            irregular = [v for v in values if v not in allowed_binary]
            if irregular:
                irregular_dict["binary"][col] = irregular

        # ---- CATEGORICAL ----
        for col in result["categorical"]:
            values = df[col].dropna().unique()
            irregular_dict["categorical"][col] = list(values)  # for categorical, report all values

        # ---- CONTINUOUS ----
        for col in result["continuous"]:
            values = df[col].dropna().unique()
            # check if all values are numeric
            irregular = [v for v in values if not isinstance(v, (int, float))]
            if irregular:
                irregular_dict["continuous"][col] = irregular

        # ---- ORDINAL ----
        for col in result["ordinal"]:
            values = df[col].dropna().unique()
            irregular_dict["ordinal"][col] = list(values)  # same as categorical for now

        # ---- PRINT SUMMARY ----
        print("\n====== IRREGULAR / UNEXPECTED ENTRIES ======\n")
        for cat, cols in irregular_dict.items():
            print(f"--- {cat.upper()} ---")
            if not cols:
                print("No irregular entries found.\n")
                continue
            for col, vals in cols.items():
                print(f"{col}: {vals}")
            print("\n")

        return irregular_dict

    def clean_irregular_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert medication values to boolean."""
        def convert_binary_values(value):
            true_values = {'1', '1: Yes', '1.0', 1.0}
            false_values = {'0', '0: No', '0.0', 0.0}
            
            if value in true_values:
                return 1
            elif value in false_values:
                return 0
            else:
                return np.nan

        # Columns to exclude from conversion
        exclude_columns = ['ID', 'Visit']
        
        # List of variables that indicate binary columns
        binary_vars = self.binary_vars  # assuming you store them in the class

        df_copy = df.copy()

        # --- NEW MASK ---
        binary_mask = [
            (col not in exclude_columns) and
            any(b in col for b in binary_vars)
            for col in df_copy.columns
        ]
        binary_mask = pd.Index(binary_mask)  # convert to pandas mask

        # Apply elementwise conversion only to selected columns
        for col in df_copy.columns[binary_mask]:
            df_copy[col] = df_copy[col].map(convert_binary_values)

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

    def save_processed_data(self, output_dir: str = "./"):
        """Save processed dataframes to CSV files."""
        if self.clinical_df is not None:
            self.clinical_df.to_csv(os.path.join(output_dir, "clinical_variable_df.csv"), index=False)
        
        if self.kxr_df is not None:
            self.kxr_df.to_csv(os.path.join(output_dir, "kxr_df.csv"), index=False)
        
        if self.outcomes_df is not None:
            self.outcomes_df.to_csv(os.path.join(output_dir, "outcomes.csv"), index=False)
            
        if self.enrollees_df is not None:
            self.enrollees_df.to_csv(os.path.join(output_dir, "enrollees.csv"), index=False)

        if self.all_visits_df is not None:
            self.all_visits_df.to_csv(os.path.join(output_dir, "all_visits_df.csv"), index=False)        
        
        if self.baseline_df is not None:
            self.baseline_df.to_csv(os.path.join(output_dir, "baseline_df.csv"), index=False)

        if self.all_visits_clean_df is not None:
            self.all_visits_clean_df.to_csv(os.path.join(output_dir, "all_visits_clean_df.csv"), index=False)

        if self.all_visits_clean_simulation_df is not None:
            self.all_visits_clean_simulation_df.to_csv(os.path.join(output_dir, "all_visits_clean_simulation_df.csv"), index=True)    

    def save_tensors(self, output_dir: str = "./"):
        """Save processed NxTxF input tensor."""
        if self.input_tensor_df is not None:
            path = os.path.join(output_dir, "input_tensor.parquet")
            self.input_tensor_df.to_parquet(path)
            print(f"Saved input tensor to {path}")
        
        if self.input_mask_df is not None:
            path = os.path.join(output_dir, "input_mask.parquet")
            self.input_mask_df.to_parquet(path)
            print(f"Saved input mask to {path}")

        if self.target_tensor_df is not None:
            path = os.path.join(output_dir, "target_tensor.parquet")
            self.target_tensor_df.to_parquet(path)
            print(f"Saved target tensor to {path}") 

        if self.target_mask_df is not None:
            path = os.path.join(output_dir, "target_mask.parquet")
            self.target_mask_df.to_parquet(path)
            print(f"Saved target mask to {path}")
    
    def read_processed_data(self, input_dir: str = "./"):
        """Read processed dataframes from CSV files."""

        # Initialize all variables as None
        clinical_df = kxr_df = outcomes_df = enrollees_df = all_visits_df = baseline_df = all_visits_clean_df = all_visits_clean_simulation_df = None

        # Clinical
        clinical_path = os.path.join(input_dir, "clinical_variable_df.csv")
        if os.path.exists(clinical_path):
            clinical_df = pd.read_csv(clinical_path, delimiter = ',', low_memory=False)
            self.clinical_df = clinical_df

        # KXR
        kxr_path = os.path.join(input_dir, "kxr_df.csv")
        if os.path.exists(kxr_path):
            kxr_df = pd.read_csv(kxr_path, delimiter = ',', low_memory=False)
            self.kxr_df = kxr_df

        # Outcomes
        outcomes_path = os.path.join(input_dir, "outcomes.csv")
        if os.path.exists(outcomes_path):
            outcomes_df = pd.read_csv(outcomes_path, delimiter = ',', low_memory=False)
            self.outcomes_df = outcomes_df

        # Enrollees
        enrollees_path = os.path.join(input_dir, "enrollees.csv")
        if os.path.exists(enrollees_path):
            enrollees_df = pd.read_csv(enrollees_path, delimiter = ',', low_memory=False)
            self.enrollees_df = enrollees_df

        # All visits
        all_visits_path = os.path.join(input_dir, "all_visits_df.csv")
        if os.path.exists(all_visits_path):
            all_visits_df = pd.read_csv(all_visits_path, delimiter = ',', low_memory=False)
            self.all_visits_df = all_visits_df

        # Baseline
        baseline_path = os.path.join(input_dir, "baseline_df.csv")
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path, delimiter = ',', low_memory=False)
            self.baseline_df = baseline_df

        # Pain trajectory clusters
        pain_trajectory_clusters_path = os.path.join(input_dir, "all_visits_pain_trajectory_clusters.csv")
        if os.path.exists(pain_trajectory_clusters_path):
            all_visits_pain_trajectory_clusters = None #pd.read_csv(pain_trajectory_clusters_path, delimiter = ',', low_memory=False)
            self.all_visits_pain_trajectory_clusters = all_visits_pain_trajectory_clusters

        # Clean all visits
        clean_all_visits_path = os.path.join(input_dir, "all_visits_clean_df.csv")
        if os.path.exists(clean_all_visits_path):
            all_visits_clean_df = pd.read_csv(clean_all_visits_path, delimiter = ',', low_memory=False)
            self.all_visits_clean_df = all_visits_clean_df

        # Clean simulation all visits
        clean_simulation_all_visits_path = os.path.join(input_dir, "all_visits_clean_simulation_df.csv")
        if os.path.exists(clean_simulation_all_visits_path):
            all_visits_clean_simulation_df = pd.read_csv(clean_simulation_all_visits_path, delimiter = ',', low_memory=False, index_col=['ID', 'Visit'])
            self.all_visits_clean_simulation_df = all_visits_clean_simulation_df    

        return clinical_df, kxr_df, outcomes_df, enrollees_df, all_visits_df, baseline_df, all_visits_pain_trajectory_clusters, all_visits_clean_df, all_visits_clean_simulation_df

    def load_input_tensor(self, input_dir: str = "./"):
        """Load processed NxTxF input tensor."""
        path = os.path.join(input_dir, "input_tensor.parquet")
        self.input_tensor_df = pd.read_parquet(path)
        self.input_tensor_df = self.input_tensor_df.sort_index()
        print(f"Loaded input tensor from {path}")
