import numpy as np
import pandas as pd


class DataDecAnalyst:
    def numeric_var(self):
        data2 = self.num_data.values
        numeric_var_dec = np.empty((22, len(self.num_name)), dtype=np.object)
        for i in range(len(self.num_name)):
            temp_data = data2[:, i]
            temp_data = temp_data.astype(float)
            numeric_var_dec[0, i] = self.num_name[i]
            numeric_var_dec[1, i] = len(temp_data)
            numeric_var_dec[2, i] = np.sum(np.isnan(temp_data))
            temp_data = temp_data[~np.isnan(temp_data)]
            numeric_var_dec[3, i] = np.sum(temp_data < 0)
            numeric_var_dec[4, i] = np.sum(temp_data == 0)
            numeric_var_dec[5, i] = np.sum(temp_data > 0)
            numeric_var_dec[6, i] = numeric_var_dec[2, i] / numeric_var_dec[1, i]
            numeric_var_dec[7, i] = numeric_var_dec[3, i] / numeric_var_dec[1, i]
            numeric_var_dec[8, i] = numeric_var_dec[4, i] / numeric_var_dec[1, i]

            numeric_var_dec[9, i] = np.min(temp_data)
            numeric_var_dec[10, i] = np.max(temp_data)
            numeric_var_dec[11, i] = np.mean(temp_data)
            numeric_var_dec[12, i] = np.std(temp_data)
            numeric_var_dec[13, i] = np.percentile(temp_data, 1)
            numeric_var_dec[14, i] = np.percentile(temp_data, 5)
            numeric_var_dec[15, i] = np.percentile(temp_data, 10)
            numeric_var_dec[16, i] = np.percentile(temp_data, 25)
            numeric_var_dec[17, i] = np.percentile(temp_data, 50)
            numeric_var_dec[18, i] = np.percentile(temp_data, 75)
            numeric_var_dec[19, i] = np.percentile(temp_data, 90)
            numeric_var_dec[20, i] = np.percentile(temp_data, 95)
            numeric_var_dec[21, i] = np.percentile(temp_data, 99)
        stat_name = [
            "VarName",
            "Obs",
            "Missobs",
            "NegaObs",
            "ZeroObs",
            "PosiObs",
            "MissRatio",
            "NegaRatio",
            "ZeroRatio",
            "MinValue",
            "MaxValue",
            "MeanValue",
            "StdDev",
            "P1",
            "P5",
            "P10",
            "P25",
            "P50",
            "P75",
            "P90",
            "P95",
            "P99",
        ]
        numeric_dec = pd.DataFrame(numeric_var_dec.T, columns=stat_name)
        try:
            numeric_dec.to_csv('step' + self.step + 'NumDec.csv', encoding='ansi', index=False)
        except:
            numeric_dec.to_csv('step' + self.step + 'NumDec.csv',  index=False)
    def str_var(self):
        data2 = self.str_data.values
        for i in range(len(self.str_name)):
            temp_data = data2[:, i].astype(str)
            class_dum = np.unique(temp_data)
            len_class = len(temp_data)
            for j in range(len(class_dum)):
                class_obs = np.sum(temp_data == class_dum[j])
                class_ratio = class_obs / len_class
                if i == 0 & j == 0:
                    str_var_dec = np.array(
                        [self.str_name[i], class_dum[j], class_obs, class_ratio]
                    )
                else:
                    temp = np.array(
                        [self.str_name[i], class_dum[j], class_obs, class_ratio]
                    )
                    str_var_dec = np.vstack((str_var_dec, temp))
        view_data = pd.DataFrame(
            str_var_dec, columns=["VarName", "ClassName", "ClassObs", "ClassRatio"]
        )
        try:
            view_data.to_csv('step' + self.step + 'StrDec.csv', encoding='ansi', index=False)
        except:
            view_data.to_csv('step' + self.step + 'StrDec.csv',  index=False)
    def __init__(self, str_name: list, num_name: list, data: pd.DataFrame, step: str = ''):
        self.step = step
        self.str_name = str_name
        self.num_name = num_name
        self.str_data = data[self.str_name]
        self.num_data = data[self.num_name]
        if len(num_name) > 0:
            self.numeric_var()
        if len(str_name) > 0:
            self.str_var()
