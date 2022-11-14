#### Usage: python -W ignore 02_2.geneExp_Predictor.py 0 100 0 0 40000
########################################################################################################################
# Function: Predict gene expression of immune cells in tumor from gene expression of immune cells in blood and clinical
#           variables using ElasticNet regression.
# Params:
#   Para.1: random seed  [0..9]
#   Para.2: resampling number
#   Para.3: cell type index  [0..11]
#   Para.4: confounder index  [0..6]
#   Para.5: predict the first N genes
# Input files:
#   ./input/PBMC_clinical_info.csv
#   ./input/allGenes_PBMC_[cellType].csv
#   ./input/expressedGenes_Tumor_[cellType].csv
# Output files:
#   ./output/GeneExpPred_[cellType]_[confounderType]_randSeed[Para.1]_resNUM[Para.2].txt
# Note: to continue following analysis, one should exhaust all combination of Para.1, Para.3 and Para.4 to obtain all
#       the result files.
########################################################################################################################

import os
from sklearn.linear_model import ElasticNet
import random
import numpy as np
import pandas as pd
import sys
import time
from sklearn.preprocessing import StandardScaler
from scipy import stats

StartTime = time.time()

randSeed = int(sys.argv[1]) # 1
random.seed(randSeed)
np.random.seed(randSeed)
resampleNUM = int(sys.argv[2])
LASSO_alpha = 0.001
CellType_ind = int(sys.argv[3]) # cell type name index of CellNames_machine: 0-11
conF_ind = int(sys.argv[4])  # co-founder index: 0-6
confounder_names = ['None', 'Alcohol', 'Sex', 'Age', 'Tissue', 'HPV', 'Tobacco']
tested_geneNUM = int(sys.argv[5]) # in total 33694 genes

CellNames_human = ['Cycling T cells', 'Cytotoxic T cells', 'DC',
                     'Germinal center B cells', 'Helper T cells', 'Macrophages',
                     'Memory B cells', 'Monocytes', 'Naive B cells', 'NK cells',
                     'Plasma cells', 'Regulatory T cells'] # All immune cell types
CellNames_machine = [c.replace('+', '.plus.') for c in CellNames_human]
CellNames_machine = [c.replace('/', '.slash.') for c in CellNames_machine]
CellNames_machine = [c.replace(' ', '.space.') for c in CellNames_machine]
CellNames_machine = [c.replace('-', '.dash.') for c in CellNames_machine]
TotalCellNumber = len(CellNames_machine)

obs_sample_size_min = 20 # if samples/observations for a cell type are < 20, the cell type will not be analysed
tested_cellNUM = len(CellNames_machine) # total cell types
ElesticNet_max_iter = 1000
LASSO_ratio = 0.5

paired_Tumor_PBMC_dict = {'Cycling.space.T.space.cells': 'Cycling.space.T.space.cells',
                        'Cytotoxic.space.T.space.cells': 'Cytotoxic.space.T.space.cells',
                        'DC': 'DC',
                        'Helper.space.T.space.cells': 'Helper.space.T.space.cells',
                        'Macrophages': 'Monocytes',
                        'Memory.space.B.space.cells': 'Memory.space.B.space.cells',
                        'Monocytes': 'Monocytes',
                        'Naive.space.B.space.cells': 'Naive.space.B.space.cells',
                        'NK.space.cells': 'NK.space.cells',
                        'Plasma.space.cells': 'Plasma.space.cells',
                        'Regulatory.space.T.space.cells': 'Regulatory.space.T.space.cells',
                        } # use the most predictive cell types in blood
fnOutNA = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt'%(CellNames_machine[CellType_ind], confounder_names[conF_ind], randSeed, resampleNUM)

effective_sample_NA = ['HNSCC_P1', 'HNSCC_P2', 'HNSCC_P3', 'HNSCC_P4', 'HNSCC_P6', 'HNSCC_P7',
                       'HNSCC_P8', 'HNSCC_P9', 'HNSCC_P10', 'HNSCC_P11', 'HNSCC_P12', 'HNSCC_P13', 'HNSCC_P14',
                       'HNSCC_P15', 'HNSCC_P16', 'HNSCC_P17', 'HNSCC_P18', 'HNSCC_P19', 'HNSCC_P20',
                       'HNSCC_P21', 'HNSCC_P22', 'HNSCC_P23', 'HNSCC_P24', 'HNSCC_P25', 'HNSCC_P26'] # 'HNSCC_P5' is excluded as an outlier

print("Read in files ...")
data_x1_fn = './input/PBMC_clinical_info.csv'
x1_raw = pd.read_csv(data_x1_fn, index_col=0, keep_default_na=False, na_values=['NAN'])
x1_raw = x1_raw.loc[effective_sample_NA]
cellAbundance_df = x1_raw.iloc[:,0:TotalCellNumber]
hpv_df = pd.get_dummies(x1_raw.hpv_status)
hpv_df = hpv_df[['HPV_positive']]
sex_df = pd.get_dummies(x1_raw.Sex)
sex_df = sex_df[['Male']]
age_df = x1_raw[['Age']]
tissue_df = pd.get_dummies(x1_raw.Tumorsite)
tissue_df = tissue_df.rename(columns=dict(zip(list(tissue_df.columns),['Tissue_'+c for c in list(tissue_df.columns)])))
tobacco_df = pd.get_dummies(x1_raw.TobaccoUse)
tobacco_df = tobacco_df.rename(columns=dict(zip(list(tobacco_df.columns),['Tobacco_'+c for c in list(tobacco_df.columns)])))
alcohol_df = pd.get_dummies(x1_raw.AlcoholUse)
alcohol_df = alcohol_df.rename(columns=dict(zip(list(alcohol_df.columns),['Alcohol_'+c for c in list(alcohol_df.columns)])))
alcohol_df.drop(['Alcohol_Unknown'], inplace=True, axis=1)

breakPoint_list = [0,5,10,15,20,25]
train_data_all = []
test_data_all = []
train_predict_all = []
test_predict_all = []

x_raw_dict = {}
y_raw_dict = {}
for cell_i in range(tested_cellNUM):
    cn = CellNames_machine[cell_i]
    data_x_fn = './input/allGenes_PBMC_%s.csv' % (cn)  # allGenes_, HVG_
    if os.path.exists(data_x_fn):
        x_raw = pd.read_csv(data_x_fn, index_col=0).T  # columns are all genes, rows are samples
        x_raw_dict[cn] = x_raw
    data_y_fn = './input/expressedGenes_Tumor_%s.csv' % (cn)
    if os.path.exists(data_y_fn):
        y_raw = pd.read_csv(data_y_fn, index_col=0).T  # columns are HVGs, rows are samples
        y_geneNA_all = y_raw.columns
        y_1gene = y_raw[[y_geneNA_all[0]]]
        y_raw_dict[cn]=[y_raw, y_geneNA_all]
Tumor_cn_in_predict = list(y_raw_dict.keys())
PBMC_cn_in_predict = list(x_raw_dict.keys())

if 0:
    print('Identify the right cell type in blood for each tumor immune cell type ...')
    pcc_resample_all = [[[np.nan for _ in range(resampleNUM)] for _ in range(len(PBMC_cn_in_predict))] for _ in range(len(Tumor_cn_in_predict))]
    for res_k in range(resampleNUM):
        sample_ind = random.choices(range(len(effective_sample_NA)),k=len(effective_sample_NA))
        for cn_i in range(len(Tumor_cn_in_predict)):
            data_t = y_raw_dict[Tumor_cn_in_predict[cn_i]].iloc[sample_ind,]
            data_t = data_t.to_numpy().flatten() # make flat
            for cn_j in range(len(PBMC_cn_in_predict)):
                data_b = x_raw_dict[PBMC_cn_in_predict[cn_j]].iloc[sample_ind,]
                data_b = data_b.to_numpy().flatten() # make flat
                pcc_temp = ma.corrcoef(ma.masked_invalid(data_t), ma.masked_invalid(data_b))[1,0]
                pcc_resample_all[cn_i][cn_j][res_k] = pcc_temp
    fnOut_corr = open('./output/Tumor_PBMC_geneExp_CellCorr.txt','w')
    content_best = []
    for cn_i in range(len(Tumor_cn_in_predict)):
        cn_tumor = Tumor_cn_in_predict[cn_i]
        content_list= []
        pcc_mean_temp_list = []
        pcc_self = -1
        ind_self = -1
        for cn_j in range(len(PBMC_cn_in_predict)):
            cn_PBMC = PBMC_cn_in_predict[cn_j]
            pcc_mean = np.mean(pcc_resample_all[cn_i][cn_j])
            pcc_std = np.std(pcc_resample_all[cn_i][cn_j])
            if cn_PBMC == cn_tumor:
                pcc_self = pcc_mean
                ind_self = cn_j
            pcc_mean_temp_list.append(pcc_mean)
            content_list.append([cn_tumor, cn_PBMC, str(pcc_mean), str(pcc_std)] + [str(c) for c in pcc_resample_all[cn_i][cn_j]])
        pcc_mean_temp_list_sorted_ind = np.argsort(pcc_mean_temp_list)[::-1]
        if pcc_self > pcc_mean_temp_list[pcc_mean_temp_list_sorted_ind[0]] * 0.9:
            content_best.append(content_list[ind_self])
        else:
            content_best.append(content_list[pcc_mean_temp_list_sorted_ind[0]])
        content_list = [content_list[x] for x in pcc_mean_temp_list_sorted_ind]
        for cn_j in range(len(PBMC_cn_in_predict)):
            fnOut_corr.write('\t'.join(content_list[cn_j]) + '\n')
    fnOut_corr.write('\n')
    for cn_i in range(len(Tumor_cn_in_predict)):
        fnOut_corr.write('\t'.join(content_best[cn_i]) + '\n')
        print(content_best[cn_i][0:5])
    fnOut_corr.close()


print('Build Elastic regression models to predict tumor immune cell gene expression ...')
confounders_list = []
confounders_list.append('None')
confounders_list.append(alcohol_df)
confounders_list.append(sex_df)
confounders_list.append(age_df)
confounders_list.append(tissue_df)
confounders_list.append(hpv_df)
confounders_list.append(tobacco_df)


confounder = confounders_list[conF_ind]
conF_NA = confounder_names[conF_ind]
cn = CellNames_machine[CellType_ind]
geneNUM = min(tested_geneNUM, len(y_raw_dict[cn][1]))
final_result = [[[] for _ in range(geneNUM)], [[] for _ in range(geneNUM)],
                [[] for _ in range(geneNUM)], [[] for _ in range(geneNUM)],
                [[] for _ in range(geneNUM)], [[] for _ in range(geneNUM)],
                [[] for _ in range(geneNUM)], [[] for _ in range(geneNUM)],
                [[] for _ in range(geneNUM)], [[] for _ in range(geneNUM)]
               ]

samples_ind = list(range(len(effective_sample_NA)))
random.seed(randSeed)
y_raw = y_raw_dict[cn][0] # data frame
y_geneNA_all = y_raw_dict[cn][1] # list of gene names

for resample_i in range(resampleNUM):
    random.shuffle(samples_ind)  # randomly shuffle sample index to assign different training/test sets
    if cn not in Tumor_cn_in_predict:
        continue
    for gene_i in range(geneNUM): # len(y_raw.shape[0])
        y_train_true_all = []
        y_train_pred_all = []
        y_test_true_all = []
        y_test_pred_all = []
        geneNA = y_geneNA_all[gene_i]
        y_1gene = y_raw[[geneNA]]
        x_1gene = pd.DataFrame()
        cell_temp = paired_Tumor_PBMC_dict[cn]
        x_1gene[cell_temp + '_' + geneNA] = x_raw_dict[cell_temp][geneNA]
        if isinstance(confounder, pd.DataFrame): # confounder dataframe
            conF_df = confounder
        else:
            if confounder != 'None': # cell type name
                conF_df = x_raw_dict[confounder][geneNA]
            else: # None
                conF_df = pd.DataFrame()
        x_all = pd.concat([x_1gene, conF_df], axis=1)
        if x_all.empty: # x_all is empty
            continue
        for pt_i in range(1,2):
            x_test_ind = samples_ind[breakPoint_list[pt_i-1]:breakPoint_list[pt_i]]
            x_train_ind = samples_ind[0:breakPoint_list[pt_i-1]] + samples_ind[breakPoint_list[pt_i]:]
            x_train = x_all.iloc[x_train_ind]
            y_train = y_1gene.iloc[x_train_ind]
            x_test = x_all.iloc[x_test_ind]
            y_test = y_1gene.iloc[x_test_ind]
            # Data scaling
            scaler_sd = StandardScaler()
            train_sample_name = x_train.index
            test_sample_name = x_test.index
            scaler_sd.fit(x_train)
            x_train = pd.DataFrame(scaler_sd.transform(x_train))
            x_train = x_train.rename(columns=dict(zip(x_train.columns,x_all.columns)))
            x_train = x_train.rename(index=dict(zip(x_train.index, train_sample_name)))
            x_test = pd.DataFrame(scaler_sd.transform(x_test))
            x_test = x_test.rename(columns=dict(zip(x_test.columns,x_all.columns)))
            x_test = x_test.rename(index=dict(zip(x_test.index, test_sample_name)))
            train_num = len(x_train_ind)
            feature_used_temp = x_train.shape[1]
            data_train = pd.concat([y_train, x_train],axis=1)
            data_train = data_train.dropna(axis=0) # remove NAs
            y_train = data_train.iloc[:,0].tolist()
            x_train = data_train.iloc[:, 1:]
            data_test = pd.concat([y_test, x_test], axis=1)
            data_test = data_test.dropna(axis=0)  # remove NAs
            y_test = data_test.iloc[:, 0].tolist()
            x_test = data_test.iloc[:, 1:]
            if len(y_test)<3: # or len(y_train)<3   too few valid data to calculate PCC (when using all cell types, different missing values in x will crash the model)
                continue
            clf = ElasticNet(l1_ratio=LASSO_ratio, max_iter=ElesticNet_max_iter, alpha=LASSO_alpha,
                             selection='cyclic', random_state=randSeed).fit(x_train, y_train)
            y_train_pred = clf.predict(x_train)
            y_test_pred = clf.predict(x_test)
            y_train_true_all.extend(y_train)
            y_train_pred_all.extend(y_train_pred)
            y_test_true_all.extend(y_test)
            y_test_pred_all.extend(y_test_pred)
        vec_len = x_train.shape[1]
        try:
            coef_fit = clf.coef_
            interc_fit = clf.intercept_
            scaler_mean_list = list(scaler_sd.mean_)
            scaler_sd_list = list(scaler_sd.scale_)
            PCC_train, pval_train = stats.pearsonr(y_train_true_all, y_train_pred_all)
            MAE_train = np.mean(abs(np.array(y_train_true_all) - y_train_pred_all))
            NMAE_train = MAE_train/(max(y_train_true_all) - min(y_train_true_all) + 0.00000001)
            NRMSE_train = NMAE_train
            PCC_test, pval_test = stats.pearsonr(y_test_true_all,y_test_pred_all)
            MAE_test = np.mean(abs(np.array(y_test_true_all) - y_test_pred_all))
            NMAE_test = MAE_test / (max(y_test_true_all) - min(y_test_true_all) + 0.00000001)
            NRMSE_test = NMAE_test
        except:
            coef_fit = [np.nan] * vec_len
            interc_fit = np.nan
            scaler_mean_list = [np.nan] * vec_len
            scaler_sd_list = [np.nan] * vec_len
            PCC_train=np.nan
            pval_train=np.nan
            RMSE_train=np.nan
            NRMSE_train=np.nan
            PCC_test=np.nan
            pval_test=np.nan
            RMSE_test=np.nan
            NRMSE_test=np.nan
        final_result[0][gene_i].append(PCC_train)
        final_result[1][gene_i].append(PCC_test)
        final_result[2][gene_i].append(pval_train)
        final_result[3][gene_i].append(pval_test)
        final_result[4][gene_i].append(NRMSE_train)
        final_result[5][gene_i].append(NRMSE_test)
        final_result[6][gene_i].append(interc_fit)
        final_result[7][gene_i].extend(coef_fit)
        final_result[8][gene_i].extend(scaler_mean_list)
        final_result[9][gene_i].extend(scaler_sd_list)

print('Saving result to file ...')
fnOut = open(fnOutNA, 'w', buffering=1)
for gene_i in range(geneNUM):
    gene = y_geneNA_all[gene_i]
    content = sum([final_result[ii][gene_i] for ii in range(10)], [])
    content = [str(c) for c in content]
    fnOut.write(gene+'\t'+'\t'.join(content)+'\n')
fnOut.close()

EndTime = time.time()
print('All done! Time used: %.2f'%(EndTime - StartTime))