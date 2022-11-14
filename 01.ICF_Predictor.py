#### Usage: python -W ignore 01.ICF_Predictor.py 1 1000 0.001
########################################################################################################################
# Function: Predict tumor immune cell fractions (ICFs) from blood ICFs and clinical variables using ElasticNet
#           regression.
# Params:
#   Para.1: random seed
#   Para.2: resampling number
#   Para.3: LASSO_alpha
# Input files:
#   ./input/PBMC_clinical_info.csv
#   ./input/PBMC_clinical_info_trainAndTest.csv
#   ./input/TIL_info.csv
# Output files:
#   ./output/ICF_Predicted_TestData.csv
#   ./output/ICF_predictability_TrainData_stats.txt
#   ./output/ICF_trueVSprediction_TestData_[cellType].png (11 files)
########################################################################################################################

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import pandas as pd
from scipy import stats
import sys
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

StartTime = time.time()

randSeed = int(sys.argv[1]) # 1
random.seed(randSeed)
resampleNUM = int(sys.argv[2]) # 1000
LASSO_alpha = float(sys.argv[3]) # 0.001
l1_ratio = 0.5 # l1 ratio in the ElasticNet regression model
re_cal_cellType_mapDict = 0
PCC_cutoff = 0.3
NMAE_cutoff = 1
self_PCC_coef = 0.9 # If PCC between a cell type in tumor and itself in blood > 0.9*PCCmax, use itself as the most predictive cell type

print('Read in files ...')
data_x_fn = './input/PBMC_clinical_info.csv'
data_y_fn = './input/TIL_info.csv'
x_raw0 = pd.read_csv(data_x_fn, index_col=0, keep_default_na=False, na_values=['NAN'])
y_raw0 = pd.read_csv(data_y_fn, index_col=0, keep_default_na=False, na_values=['NAN'])

CellNames_full = ['Cycling.space.T.space.cells', 'Cytotoxic.space.T.space.cells', 'DC', 'Mast.space.cells',
                   'Germinal.space.center.space.B.space.cells', 'Helper.space.T.space.cells', 'Macrophages',
                   'Memory.space.B.space.cells', 'Monocytes', 'Naive.space.B.space.cells', 'NK.space.cells',
                   'Plasma.space.cells', 'Regulatory.space.T.space.cells',
                   'Regulatory.space.T.space.cells to Cytotoxic.space.T.space.cells ratio']
CellNames_human = ['Cycling T cells', 'Cytotoxic T cells', 'DC', 'Mast cells',
                   'Germinal center B cells', 'Helper T cells', 'Macrophages',
                   'Memory B cells', 'Monocytes', 'Naive B cells', 'NK cells',
                   'Plasma cells', 'Regulatory T cells',
                    'Regulatory T cells to cytotoxic T cells ratio']
CellNames_short = ['CyclingT', 'CD8T', 'DC', 'Mast', 'GCB', 'HelpT', 'Macrophages', 'MemoryB', 'Monocytes', 'NaiveB',
                    'NK', 'Plasma', 'RegT', 'Treg/CD8T']
ExcludedCellNames_PBMC = ['Cycling.space.T.space.cells', 'Mast.space.cells', 'Germinal.space.center.space.B.space.cells', 'Macrophages'] # 0,3,5
ExcludedIdx_PBMC = [0,3,4,6]
ExcludedCellNames_tumor = ['Germinal.space.center.space.B.space.cells', 'Mast.space.cells']
cellTypeNUM = len(CellNames_full)
CellNames_tumor_short = [c+'_t' for c in CellNames_short]
CellNames_PBMC_short = [c+'_b' for c in CellNames_short]
nameMap_tumor = dict(zip(CellNames_full, CellNames_tumor_short))
nameMap_PBMC = dict(zip(CellNames_full, CellNames_PBMC_short))
y_raw0 = y_raw0.rename(columns=nameMap_tumor)
x_raw0 = x_raw0.rename(columns=nameMap_PBMC)


print('Predict tumor ICF ...')
TrainSampleNames_effective = ['HNSCC_P1', 'HNSCC_P2', 'HNSCC_P3', 'HNSCC_P4', 'HNSCC_P6', 'HNSCC_P7',
                       'HNSCC_P8', 'HNSCC_P9', 'HNSCC_P10', 'HNSCC_P11', 'HNSCC_P12', 'HNSCC_P13', 'HNSCC_P14',
                       'HNSCC_P15', 'HNSCC_P16', 'HNSCC_P17', 'HNSCC_P18', 'HNSCC_P19', 'HNSCC_P20',
                       'HNSCC_P21', 'HNSCC_P22', 'HNSCC_P23', 'HNSCC_P24', 'HNSCC_P25', 'HNSCC_P26'] # 'HNSCC_P5' is excluded as an outlier
TestSampleNames = ['HNSCC2_P18', 'HNSCC2_P23', 'HNSCC2_P24', 'HNSCC2_P29', 'HNSCC2_P32']

x_raw = x_raw0.loc[TrainSampleNames_effective,]
y_raw = y_raw0.loc[TrainSampleNames_effective,] # tissue
row_num = len(x_raw.iloc[:,1])
cellAbundance_df = x_raw.iloc[:,0:cellTypeNUM-1] # PBMC
cellAbundance_df = cellAbundance_df[CellNames_PBMC_short[0:-1]]
y_raw['Treg/CD8T_t'] = y_raw.loc[:, 'RegT_t']  / (y_raw.loc[:, 'CD8T_t'] + 0.000001)
cellAbundance_df['Treg/CD8T_b'] = cellAbundance_df.loc[:, 'RegT_b']  / (cellAbundance_df.loc[:, 'CD8T_b'] + 0.000001)

x_raw_test = x_raw0.loc[TestSampleNames,]
y_raw_test = y_raw0.loc[TestSampleNames,] # tissue
cellAbundance_test_df = x_raw_test.iloc[:,0:cellTypeNUM-1] # PBMC
cellAbundance_test_df = cellAbundance_test_df[CellNames_PBMC_short[0:-1]]
y_raw_test['Treg/CD8T_t'] = y_raw_test.loc[:, 'RegT_t']  / (y_raw_test.loc[:, 'CD8T_t'] + 0.000001)
cellAbundance_test_df['Treg/CD8T_b'] = cellAbundance_test_df.loc[:, 'RegT_b']  / (cellAbundance_test_df.loc[:, 'CD8T_b'] + 0.000001)

#### Find the most predictive cell type in the blood for each cell type in the tumor
if re_cal_cellType_mapDict:
    pcc_resample_all = [[[np.nan for _ in range(resampleNUM)] for _ in range(cellTypeNUM)] for _ in range(cellTypeNUM)]
    cellType_mapDict = {}
    for cell_i in range(cellTypeNUM):
        cellNA = CellNames_tumor_short[cell_i]
        pcc_all = []
        for bs_i in range(resampleNUM):
            row_ind = random.choices(range(row_num), k=int(row_num*2/3)) # randomly choose 2/3 samples with replacement
            samples_chosen = [TrainSampleNames_effective[c] for c in row_ind]
            y = y_raw.loc[samples_chosen,cellNA]
            x = cellAbundance_df.loc[samples_chosen,:]
            pcc_temp = list(x.corrwith(y, method='pearson'))
            pcc_temp[-1] = 0 # do not consider CD8T/Treg
            pcc_all.append(pcc_temp)
            for cell_j in range(cellTypeNUM):
                pcc_resample_all[cell_i][cell_j][bs_i] = pcc_temp[cell_j]
        pcc_all = list(zip(*pcc_all))
        pcc_mean = [np.nanmedian(c) for c in pcc_all]
        pcc_mean_std = [[round(np.nanmedian(c), 3), round(np.nanstd(c), 3)] for c in pcc_all]
        pcc_self = pcc_mean[cell_i]
        sorted_ind = np.argsort(pcc_mean)[::-1]
        sorted_ind = [e for e in sorted_ind if e not in ExcludedIdx_PBMC] # remove cells not in blood
        pcc_mean_std = [[CellNames_PBMC_short[i]] + pcc_mean_std[i] for i in sorted_ind]
        pcc_max = pcc_mean_std[0][1]
        if pcc_self > pcc_max*self_PCC_coef:
            cellType_mapDict[cell_i] = CellNames_PBMC_short[cell_i]
        else:
            cellType_mapDict[cell_i] = pcc_mean_std[0][0]
else: # use pre-calculated predictive cell type mapping dict
    cellType_mapDict = {0: 'Plasma_b', 1: 'CD8T_b', 2: 'DC_b', 3: 'RegT_b', 4: 'NaiveB_b', 5: 'HelpT_b', 6: 'Plasma_b', 7: 'NaiveB_b',
                         8: 'Monocytes_b', 9: 'NaiveB_b', 10: 'Monocytes_b', 11: 'DC_b', 12: 'HelpT_b', 13: 'HelpT_b'}

##### Modeling to predict tumor immune cell relative abundance from that in the blood
conF_names = ['None', 'HPV', 'Sex', 'Age', 'Tobacco', 'Alcohol']
hpv_df = pd.get_dummies(x_raw0.hpv_status)
hpv_df = hpv_df[['HPV_positive']]
sex_df = pd.get_dummies(x_raw0.Sex)
sex_df = sex_df[['Male']]
age_df = x_raw0[['Age']]
tissue_df = pd.get_dummies(x_raw0.Tumorsite)
tissue_df = tissue_df.rename(columns=dict(zip(list(tissue_df.columns),['Tissue_'+c for c in list(tissue_df.columns)])))
tobacco_df = pd.get_dummies(x_raw0.TobaccoUse)
tobacco_df = tobacco_df.rename(columns=dict(zip(list(tobacco_df.columns),['Tobacco_'+c for c in list(tobacco_df.columns)])))
alcohol_df = pd.get_dummies(x_raw0.AlcoholUse)
alcohol_df = alcohol_df.rename(columns=dict(zip(list(alcohol_df.columns),['Alcohol_'+c for c in list(alcohol_df.columns)])))
alcohol_df.drop(['Alcohol_Unknown'], inplace=True, axis=1)
hpv_df_study1 = hpv_df.loc[TrainSampleNames_effective]
sex_df_study1 = sex_df.loc[TrainSampleNames_effective]
age_df_study1 = age_df.loc[TrainSampleNames_effective]
tissue_df_study1 = tissue_df.loc[TrainSampleNames_effective]
tobacco_df_study1 = tobacco_df.loc[TrainSampleNames_effective]
alcohol_df_study1 = alcohol_df.loc[TrainSampleNames_effective]
conFounders_list = [pd.DataFrame(), hpv_df_study1, sex_df_study1, age_df_study1, tobacco_df_study1, alcohol_df_study1]

hpv_df_study2 = hpv_df.loc[TestSampleNames]
sex_df_study2 = sex_df.loc[TestSampleNames]
age_df_study2 = age_df.loc[TestSampleNames]
tissue_df_study2 = tissue_df.loc[TestSampleNames]
tobacco_df_study2 = tobacco_df.loc[TestSampleNames]
alcohol_df_study2 = alcohol_df.loc[TestSampleNames]
conFounders_test_list = [pd.DataFrame(), hpv_df_study2, sex_df_study2, age_df_study2, tobacco_df_study2, alcohol_df_study2]

#### Training ElasticNet regression model
TrainTestBreakPoint_list = [0,5,10,15,20,25] # 26
BestFitParam_dict = {}

fnOutNA = './output/ICF_predictability_TrainData_stats.txt'
fnOut = open(fnOutNA, 'w', buffering=1)
fnOut.write('CellName\tPredictiveCellName\tBestConfounder\tPCCs\n')
for cn_i in range(cellTypeNUM):
    BestFeatureCombination = 0
    BestScore = 0
    BestMetric_list = []
    BestPCC_list = []
    BestNMAE_list = []
    DefaultScore = 0
    DefaultMetric_list = []
    cn = CellNames_tumor_short[cn_i]
    cn_human = CellNames_human[cn_i]
    cn_full_temp = CellNames_full[cn_i]
    if cn_full_temp in ExcludedCellNames_tumor:
        continue
    BestFitCoef_list = []
    BestFitInterc = 0
    BestFitScaler_mean_list = []
    BestFitScaler_sd_list = []

    RCA_name = cellType_mapDict[cn_i]
    result_dict = {}
    for conF_name in conF_names:
        result_dict[conF_name] = [[], [], [], [], [], [], [], []]
    for cof_i in range(len(conF_names)): # len(conF_names)
        conF_name = conF_names[cof_i]
        x_raw2 = pd.concat([cellAbundance_df[[RCA_name]], conFounders_list[cof_i]], axis=1)
        random.seed(randSeed)
        row_ind = list(range(row_num))
        for resample_i in range(resampleNUM):
            random.shuffle(row_ind)
            y_train_true_all = []
            y_train_pred_all = []
            y_test_true_all = []
            y_test_pred_all = []
            for pt_i in range(1,2):
                x_test_ind = row_ind[TrainTestBreakPoint_list[pt_i-1]:TrainTestBreakPoint_list[pt_i]]
                x_train_ind = row_ind[0:TrainTestBreakPoint_list[pt_i-1]] + row_ind[TrainTestBreakPoint_list[pt_i]:]
                x_train = x_raw2.iloc[x_train_ind,:]
                y_train = y_raw[cn].iloc[x_train_ind]
                x_test = x_raw2.iloc[x_test_ind,:]
                y_test = y_raw[cn].iloc[x_test_ind]
                # data scaling
                scaler_sd = StandardScaler()
                train_sample_name = x_train.index
                test_sample_name = x_test.index
                scaler_sd.fit(x_train)
                x_train = pd.DataFrame(scaler_sd.transform(x_train))
                x_train = x_train.rename(columns=dict(zip(x_train.columns,x_raw2.columns)))
                x_train = x_train.rename(index=dict(zip(x_train.index, train_sample_name)))
                x_test = pd.DataFrame(scaler_sd.transform(x_test))
                x_test = x_test.rename(columns=dict(zip(x_test.columns,x_raw2.columns)))
                x_test = x_test.rename(index=dict(zip(x_test.index, test_sample_name)))
                y_train = y_train.tolist()
                y_test = y_test.tolist()
                if LASSO_alpha > 0:
                    clf = ElasticNet(l1_ratio=l1_ratio, max_iter=1000, alpha=LASSO_alpha, selection='cyclic',
                                     random_state=randSeed).fit(x_train, y_train)
                else:
                    clf = LinearRegression().fit(x_train, y_train)
                y_train_pred = clf.predict(x_train)
                y_test_pred = clf.predict(x_test)
                y_train_true_all.extend(y_train)
                y_train_pred_all.extend(y_train_pred)
                y_test_true_all.extend(y_test)
                y_test_pred_all.extend(y_test_pred)
            coef_fit = clf.coef_
            interc_fit = clf.intercept_
            scaler_mean_list = list(scaler_sd.mean_)
            scaler_sd_list = list(scaler_sd.scale_)
            PCC_train, pval_train = stats.pearsonr(y_train_true_all, y_train_pred_all)
            MAE_train = np.mean(abs(np.array(y_train_true_all) - y_train_pred_all))
            NMAE_train = MAE_train/(np.max(y_train_true_all) - np.min(y_train_true_all) + 0.00000001)
            PCC_test, pval_test = stats.pearsonr(y_test_true_all,y_test_pred_all)
            MAE_test = np.mean(abs(np.array(y_test_true_all) - y_test_pred_all))
            NMAE_test = MAE_test/(np.max(y_test_true_all) - np.min(y_test_true_all) + 0.00000001)
            # store result
            result_dict[conF_name][0].append(PCC_train)
            result_dict[conF_name][1].append(PCC_test)
            result_dict[conF_name][2].append(pval_train)
            result_dict[conF_name][3].append(pval_test)
            result_dict[conF_name][4].append(NMAE_train)
            result_dict[conF_name][5].append(NMAE_test)
            result_dict[conF_name][6].append(list(coef_fit) + [interc_fit])
            result_dict[conF_name][7].append(scaler_mean_list + scaler_sd_list)
        PCC_mean = np.nanmedian(result_dict[conF_name][1])
        p_val_mean = np.nanmedian(result_dict[conF_name][3])
        NMAE_mean = np.nanmedian(result_dict[conF_name][5])
        score = PCC_mean
        if cof_i == 0:
            DefaultScore = score
            DefaultMetric_list = [round(PCC_mean,3), round(NMAE_mean,3), round(p_val_mean,3)]
            default_feature_comb = RCA_name + '-' + conF_name
            default_pcc_list = [x for x in result_dict[conF_name][1] if np.isnan(x) == False]
            default_NMAE_list = [x for x in result_dict[conF_name][5] if np.isnan(x) == False]
        if score > BestScore:
            BestScore = score
            BestMetric_list = [round(PCC_mean,3), round(NMAE_mean,3), round(p_val_mean,3)]
            BestFeatureCombination = RCA_name + '-' + conF_name
            BestPCC_list = [x for x in result_dict[conF_name][1] if np.isnan(x) == False]
            BestNMAE_list = [x for x in result_dict[conF_name][5] if np.isnan(x) == False]
    pval_pcc_temp = stats.median_test(BestPCC_list, default_pcc_list)[1]
    pval_NMAE_temp = stats.median_test(BestNMAE_list, default_NMAE_list)[1]
    if (pval_pcc_temp >= 0.05) and (pval_NMAE_temp >= 0.05): # no significant improvement by adding co-founder
        BestPCC_list = default_pcc_list
        BestNMAE_list = default_NMAE_list
        BestScore = DefaultScore
        BestFeatureCombination = default_feature_comb
        BestMetric_list = DefaultMetric_list
    conF_best = BestFeatureCombination.split('-')[1]
    good_coef_interc_list = list(zip(*result_dict[conF_best][6]))
    good_scaler_mean_sd_list = list(zip(*result_dict[conF_best][7]))
    # write result to file
    content = sum([result_dict[BestFeatureCombination.split('-')[1]][ii] for ii in range(8)], [])
    content = [str(c) for c in content]
    fnOut = open(fnOutNA, 'a')
    fnOut.write(cn + '\t' + RCA_name + '\t' + conF_best + '\t' + '\t'.join(content) + '\n')
    # store best fitting parameters for applying model on validation data
    BestFitCoef_list = [np.mean(c) for c in good_coef_interc_list[0:-1]]
    BestFitInterc = np.mean(good_coef_interc_list[-1])
    x_NUM = len(good_scaler_mean_sd_list)//2
    BestFitScaler_mean_list = [np.mean(c) for c in good_scaler_mean_sd_list[0:x_NUM]]
    BestFitScaler_sd_list = [np.mean(c) for c in good_scaler_mean_sd_list[x_NUM:]]
    BestFitParam_dict[cn] = [BestFeatureCombination, BestFitCoef_list, BestFitInterc, BestFitScaler_mean_list,
                             BestFitScaler_sd_list]
fnOut.close()

print('Plot tumor ICF predictability on test data ...')
plot = 1
if plot:
    ##################### Model performance on test data #####################
    for cn_i in range(cellTypeNUM):
        cn = CellNames_tumor_short[cn_i]
        cn_full_temp = CellNames_full[cn_i]
        if cn_full_temp in ExcludedCellNames_tumor:
            continue
        fit_params = BestFitParam_dict[cn]
        cn_predictor_used, co_founder_used = fit_params[0].split('-')
        BestFitCoef_list = fit_params[1]
        BestFitInterc = fit_params[2]
        BestFitScaler_mean_list = fit_params[3]
        BestFitScaler_sd_list = fit_params[4]
        x = pd.concat(
            [cellAbundance_test_df[[cn_predictor_used]], conFounders_test_list[conF_names.index(co_founder_used)]],
            axis=1)
        y = np.array(y_raw_test[cn])

        scaler_sd = StandardScaler()
        scaler_sd.fit(x)
        scaler_sd.mean_ = np.array(BestFitScaler_mean_list)
        scaler_sd.scale_ = np.array(BestFitScaler_sd_list)
        x = pd.DataFrame(scaler_sd.transform(x))

        term1 = x * BestFitCoef_list
        term1 = np.array(term1.sum(axis=1))
        y_pred = term1 + BestFitInterc
        PCC, pval = stats.pearsonr(y, y_pred)
        MAE_test = np.mean(abs(y - y_pred))
        NMAE_test = MAE_test / (np.max(y) - np.min(y) + 0.00000001)
        cn_modify = cn.replace('/', '_')
        output_fig = './output/ICF_trueVSprediction_TestData_%s.png' % (cn_modify)
        plt.figure(figsize=(3.5, 2.2))
        plt.rcParams['font.size'] = 16
        plt.subplots_adjust(left=0.35, bottom=0.25, right=0.95, top=0.95, wspace=0, hspace=0)
        ax1 = plt.subplot(111)
        ax1.scatter(y_pred, y, s=35, c='k', marker='o')
        if cn != 'CyclingT_t':
            slope, intercept, r, p, se = linregress(y_pred, y)
            rho, pval = stats.spearmanr(y_pred, y)
            ax1.plot(y_pred, intercept + slope * y_pred, 'r', linewidth=1)
            ax1.annotate('r = ' + str(round(r, 2)), xy=(0.05, 0.93), xycoords='axes fraction')
            ax1.annotate('rho = ' + str(round(rho, 2)), xy=(0.05, 0.8), xycoords='axes fraction')
        ax1.set_xlabel("Predicted", color="k")
        ax1.set_ylabel("True", color="k")
        x_1 = round(max(0, min(y_pred)), 2)
        x_2 = round(max(0, max(y_pred)), 2)
        y_1 = round(max(0, min(y)), 2)
        y_2 = round(max(0.01, max(y)), 2)
        ax1.set_xticks([x_1, x_2])
        ax1.set_yticks([y_1, y_2])
        diff_y = max(0.01, (max(y) - min(y)) * 0.1)
        diff_x = max(0.01, (max(y_pred) - min(y_pred)) * 0.1)
        ax1.set_xlim([min(y_pred) - diff_x, max(y_pred) + diff_x])
        if y_2 < 0.02:
            if y_1 < 0.001:
                ax1.set_ylim([-0.002, max(y) + diff_y * 1.1])
            else:
                ax1.set_ylim([min(y) - diff_y, max(y) + diff_y * 1.1])
        else:
            ax1.set_ylim([min(y) - diff_y, max(y) + diff_y * 3.5])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.savefig(output_fig, dpi=200)
        plt.close()

print('Apply model to predict tumor ICF from blood ...')
apply=1
if apply:
    effective_TestSample2_NA = ['HNSCC2_P01', 'HNSCC2_P02', 'HNSCC2_P04', 'HNSCC2_P05', 'HNSCC2_P08', 'HNSCC2_P09',
                               'HNSCC2_P10', 'HNSCC2_P12', 'HNSCC2_P13', 'HNSCC2_P14', 'HNSCC2_P15', 'HNSCC2_P16',
                               'HNSCC2_P17', 'HNSCC2_P18', 'HNSCC2_P19', 'HNSCC2_P20', 'HNSCC2_P21', 'HNSCC2_P22',
                               'HNSCC2_P23', 'HNSCC2_P24', 'HNSCC2_P25', 'HNSCC2_P26', 'HNSCC2_P28', 'HNSCC2_P29',
                               'HNSCC2_P30', 'HNSCC2_P31', 'HNSCC2_P32']
    data_x_fn = './input/PBMC_clinical_info_trainAndTest.csv' #
    x_raw01 = pd.read_csv(data_x_fn, index_col=0, keep_default_na=False, na_values=['NAN'])
    x_raw01 = x_raw01.rename(columns=nameMap_PBMC)
    x_raw_test2 = x_raw01.loc[effective_TestSample2_NA,]
    cellAbundance_test2_df = x_raw_test2.iloc[:,0:cellTypeNUM-1] # PBMC
    cellAbundance_test2_df = cellAbundance_test2_df[CellNames_PBMC_short[0:-1]]
    cellAbundance_test2_df['Treg/CD8T_b'] = cellAbundance_test2_df.loc[:, 'RegT_b']  / (cellAbundance_test2_df.loc[:, 'CD8T_b'] + 0.000001)
    hpv_df = pd.get_dummies(x_raw01.hpv_status)
    hpv_df = hpv_df[['HPV_positive']]
    sex_df = pd.get_dummies(x_raw01.Sex)
    sex_df = sex_df[['Male']]
    age_df = x_raw01[['Age']]
    tissue_df = pd.get_dummies(x_raw01.Tumorsite)
    tissue_df = tissue_df.rename(columns=dict(zip(list(tissue_df.columns),['Tissue_'+c for c in list(tissue_df.columns)])))
    tobacco_df = pd.get_dummies(x_raw01.TobaccoUse)
    tobacco_df = tobacco_df.rename(columns=dict(zip(list(tobacco_df.columns),['Tobacco_'+c for c in list(tobacco_df.columns)])))
    alcohol_df = pd.get_dummies(x_raw01.AlcoholUse)
    alcohol_df = alcohol_df.rename(columns=dict(zip(list(alcohol_df.columns),['Alcohol_'+c for c in list(alcohol_df.columns)])))
    alcohol_df.drop(['Alcohol_Unknown'], inplace=True, axis=1)
    hpv_df_study2 = hpv_df.loc[effective_TestSample2_NA]
    sex_df_study2 = sex_df.loc[effective_TestSample2_NA]
    age_df_study2 = age_df.loc[effective_TestSample2_NA]
    tissue_df_study2 = tissue_df.loc[effective_TestSample2_NA]
    tobacco_df_study2 = tobacco_df.loc[effective_TestSample2_NA]
    alcohol_df_study2 = alcohol_df.loc[effective_TestSample2_NA]
    conFounders_test2_list = [pd.DataFrame(), hpv_df_study2, sex_df_study2, age_df_study2, tobacco_df_study2, alcohol_df_study2]
    ##################### Predict RCA on test data #####################
    predicted_ICF_df = pd.DataFrame(effective_TestSample2_NA, columns=[CellNames_full[0]])
    predicted_ICF_df.index = effective_TestSample2_NA
    cellAbundance_df = cellAbundance_test2_df
    for cn_i in range(cellTypeNUM): #range(cellTypeNUM):
        cn = CellNames_tumor_short[cn_i]
        cn_full_temp = CellNames_full[cn_i]
        if cn_full_temp in ExcludedCellNames_tumor:
            predicted_ICF_df[cn_full_temp] = np.nan
            continue
        fit_params = BestFitParam_dict[cn]
        cn_predictor_used, confounder_used = fit_params[0].split('-')
        BestFitCoef_list = fit_params[1]
        BestFitInterc = fit_params[2]
        BestFitScaler_mean_list = fit_params[3]
        BestFitScaler_sd_list = fit_params[4]
        x = pd.concat([cellAbundance_df[[cn_predictor_used]], conFounders_test2_list[conF_names.index(confounder_used)]], axis=1)
        scaler_sd = StandardScaler()
        scaler_sd.fit(x)
        scaler_sd.mean_ = np.array(BestFitScaler_mean_list)
        scaler_sd.scale_ = np.array(BestFitScaler_sd_list)
        x = pd.DataFrame(scaler_sd.transform(x))
    
        term1 = x * BestFitCoef_list
        term1 = np.array(term1.sum(axis=1))
        y_pred = term1 + BestFitInterc
        predicted_ICF_df[cn_full_temp] = y_pred
    predicted_ICF_df.to_csv('./output/ICF_Predicted_TestData.csv', index=True)


EndTime = time.time()
print('All done! Time used: %.2f'%(EndTime - StartTime))