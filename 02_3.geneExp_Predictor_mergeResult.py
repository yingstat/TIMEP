#### Usage: python -W ignore 02_3.geneExp_Predictor_mergeResult.py 0.3 1 0.05 100
########################################################################################################################
# Function: Merge raw results from 02_2.geneExp_Predictor.py and do statistics on predictability of genes.
# Params:
#   Para.1: PCC cutoff 
#   Para.2: NMAE_cutoff
#   Para.3: FDR_cutoff
#   Para.4: resampling number
# Input files:
#   ./input/PBMC_clinical_info.csv
#   ./output/GeneExpPred_[cellType]_[confounderType]_randSeed[Para.1].txt
# Output files:
#   ./output/final_GeneExpPred_PCC[Para.1]_NMAE[Para.2]_FDR[Para.3]_geneList.csv
#   ./output/final_GeneExpPred_[cellType]_summary.csv
########################################################################################################################

import numpy as np
import sys
from statsmodels.stats import multitest
from scipy import stats
import time

StartTime = time.time()

PCC_cutoff = float(sys.argv[1]) #0.3
NMAE_cutoff = float(sys.argv[2]) #1
FDR_cutoff = float(sys.argv[3]) # 0.05
resampleNUM = int(sys.argv[4]) # 100
resampleNUM2 = 10*resampleNUM
conF_nameIndex_dict = {'None': 0, 'Alcohol': 1, 'Sex':2, 'Age':3, 'Tissue':4, 'HPV':5, 'Tobacco':6}
conF_columnNUM_dict = {'None': 1, 'Alcohol': 3, 'Sex':2, 'Age':2, 'Tissue':11, 'HPV':2, 'Tobacco':4} #

effective_sample_NA_study1 = ['HNSCC_P1', 'HNSCC_P2', 'HNSCC_P3', 'HNSCC_P4', 'HNSCC_P6', 'HNSCC_P7',
                       'HNSCC_P8', 'HNSCC_P9', 'HNSCC_P10', 'HNSCC_P11', 'HNSCC_P12', 'HNSCC_P13', 'HNSCC_P14',
                       'HNSCC_P15', 'HNSCC_P16', 'HNSCC_P17', 'HNSCC_P18', 'HNSCC_P19', 'HNSCC_P20',
                       'HNSCC_P21', 'HNSCC_P22', 'HNSCC_P23', 'HNSCC_P24', 'HNSCC_P25', 'HNSCC_P26']
CellNames_human = ['Memory B cells', 'Regulatory T cells', 'Helper T cells', 'Naive B cells', 'Plasma cells',
                   'Macrophages', 'DC', 'Monocytes', 'Cytotoxic T cells', 'Cycling T cells', 'NK cells'
                  ]
CellNames_machine = [c.replace('+', '.plus.') for c in CellNames_human]
CellNames_machine = [c.replace('/', '.slash.') for c in CellNames_machine]
CellNames_machine = [c.replace(' ', '.space.') for c in CellNames_machine]
CellNames_machine = [c.replace('-', '.dash.') for c in CellNames_machine]

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

confounder_names = ['None', 'Alcohol', 'Sex', 'Age', 'HPV', 'Tobacco']
confounderNUM = len(confounder_names)

############# Extract information of training parameters and performance stats.
fnOutNA3 = './output/final_GeneExpPred_PCC%s_NMAE%s_FDR%s_geneList.csv' % (sys.argv[1], sys.argv[2], sys.argv[3])
fnOut3 = open(fnOutNA3, 'w', buffering=1)
content_header0 = ['CellType_tumor', 'PredictorCellType_blood', 'Confounder', 'GeneList']
fnOut3.write(','.join(content_header0) + '\n')
panCell_mergedGeneLists = [[] for _ in range(confounderNUM+1)]
for cn in CellNames_machine:
    print('Processing cell %s ...'%cn)
    conF = confounder_names[0]
    result_list = [[] for _ in range(1+6+9*(confounderNUM-1)+7)]
    param_list = [[] for _ in range(4*confounderNUM+1)] # interc, coefs, scaler_means, scaler_sds
    fnOutNA = './output/final_GeneExpPred_%s_summary.csv' % (cn)
    fnInNA1 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 0, resampleNUM)
    data1 = open(fnInNA1, 'r').readlines()
    geneNUM = len(data1)
    geneNames_all = [c.split('\t')[0] for c in data1]
    result_list[0] = geneNames_all
    param_list[0] = geneNames_all
    geneNames_all = np.array(geneNames_all)
    for ConF_ind in range(confounderNUM):
        conF = confounder_names[ConF_ind]
        coef_scaler_paramNUM = conF_columnNUM_dict[conF]
        fnInNA1 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 0, resampleNUM)
        fnInNA2 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 1, resampleNUM)
        fnInNA3 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 2, resampleNUM)
        fnInNA4 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 3, resampleNUM)
        fnInNA5 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 4, resampleNUM)
        fnInNA6 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 5, resampleNUM)
        fnInNA7 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 6, resampleNUM)
        fnInNA8 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 7, resampleNUM)
        fnInNA9 = './output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 8, resampleNUM)
        fnInNA10 ='./output/GeneExpPred_%s_%s_randSeed%s_resNUM%s.txt' % (cn, conF, 9, resampleNUM)
        data1 = open(fnInNA1, 'r').readlines()
        data2 = open(fnInNA2, 'r').readlines()
        data3 = open(fnInNA3, 'r').readlines()
        data4 = open(fnInNA4, 'r').readlines()
        data5 = open(fnInNA5, 'r').readlines()
        data6 = open(fnInNA6, 'r').readlines()
        data7 = open(fnInNA7, 'r').readlines()
        data8 = open(fnInNA8, 'r').readlines()
        data9 = open(fnInNA9, 'r').readlines()
        data10 =open(fnInNA10,'r').readlines()
        pcc_mean_list =  [0.01 for _ in range(geneNUM)]
        pcc_pval_list =  [0.01 for _ in range(geneNUM)]
        NMAE_mean_list = [0.01 for _ in range(geneNUM)]
        NMAE_pval_list = [0.01 for _ in range(geneNUM)]
        fitting_interc_mean_list = [0 for _ in range(geneNUM)]
        fitting_coefs_mean_list = [[0 for _ in range(coef_scaler_paramNUM)] for _ in range(geneNUM)]
        scaler_means_mean_list = [[0 for _ in range(coef_scaler_paramNUM)] for _ in range(geneNUM)]
        scaler_sds_mean_list = [[0 for _ in range(coef_scaler_paramNUM)] for _ in range(geneNUM)]
        pcc_all_raw = [[0.01 for _ in range(resampleNUM2)] for _ in range(geneNUM)]  # predicted PCCs for each gene

        for line_i in range(geneNUM):
            line1_1 = data1[line_i]
            words1_1 = line1_1.strip('\n').split('\t')
            gene_temp = words1_1[0]
            words1_1 = words1_1[1:]
            line2_1 = data2[line_i]
            words2_1 = line2_1.strip('\n').split('\t')[1:]
            line3_1 = data3[line_i]
            words3_1 = line3_1.strip('\n').split('\t')[1:]
            line4_1 = data4[line_i]
            words4_1 = line4_1.strip('\n').split('\t')[1:]
            line5_1 = data5[line_i]
            words5_1 = line5_1.strip('\n').split('\t')[1:]
            line6_1 = data6[line_i]
            words6_1 = line6_1.strip('\n').split('\t')[1:]
            line7_1 = data7[line_i]
            words7_1 = line7_1.strip('\n').split('\t')[1:]
            line8_1 = data8[line_i]
            words8_1 = line8_1.strip('\n').split('\t')[1:]
            line9_1 = data9[line_i]
            words9_1 = line9_1.strip('\n').split('\t')[1:]
            line10_1 = data10[line_i]
            words10_1 = line10_1.strip('\n').split('\t')[1:]
            end_pos7 = (7 + coef_scaler_paramNUM) * resampleNUM
            end_pos8 = (7 + 2*coef_scaler_paramNUM) * resampleNUM
            end_pos9 = (7 + 3*coef_scaler_paramNUM) * resampleNUM
            words = [gene_temp] + \
                    words1_1[0:resampleNUM] + words2_1[0:resampleNUM] + words3_1[0:resampleNUM] + words4_1[0:resampleNUM] + words5_1[0:resampleNUM] + words6_1[0:resampleNUM] + words7_1[0:resampleNUM] + words8_1[0:resampleNUM] + words9_1[0:resampleNUM] + words10_1[0:resampleNUM] + \
                    words1_1[resampleNUM:(2*resampleNUM)] + words2_1[resampleNUM:(2*resampleNUM)] + words3_1[resampleNUM:(2*resampleNUM)] + words4_1[resampleNUM:(2*resampleNUM)] + words5_1[resampleNUM:(2*resampleNUM)] + words6_1[resampleNUM:(2*resampleNUM)] + words7_1[resampleNUM:(2*resampleNUM)] + words8_1[resampleNUM:(2*resampleNUM)] + words9_1[resampleNUM:(2*resampleNUM)] + words10_1[resampleNUM:(2*resampleNUM)] + \
                    words1_1[(2*resampleNUM):(3*resampleNUM)] + words2_1[(2*resampleNUM):(3*resampleNUM)] + words3_1[(2*resampleNUM):(3*resampleNUM)] + words4_1[(2*resampleNUM):(3*resampleNUM)] + words5_1[(2*resampleNUM):(3*resampleNUM)] + words6_1[(2*resampleNUM):(3*resampleNUM)] + words7_1[(2*resampleNUM):(3*resampleNUM)] + words8_1[(2*resampleNUM):(3*resampleNUM)] + words9_1[(2*resampleNUM):(3*resampleNUM)] + words10_1[(2*resampleNUM):(3*resampleNUM)] + \
                    words1_1[(3*resampleNUM):(4*resampleNUM)] + words2_1[(3*resampleNUM):(4*resampleNUM)] + words3_1[(3*resampleNUM):(4*resampleNUM)] + words4_1[(3*resampleNUM):(4*resampleNUM)] + words5_1[(3*resampleNUM):(4*resampleNUM)] + words6_1[(3*resampleNUM):(4*resampleNUM)] + words7_1[(3*resampleNUM):(4*resampleNUM)] + words8_1[(3*resampleNUM):(4*resampleNUM)] + words9_1[(3*resampleNUM):(4*resampleNUM)] + words10_1[(3*resampleNUM):(4*resampleNUM)] + \
                    words1_1[(4*resampleNUM):(5*resampleNUM)] + words2_1[(4*resampleNUM):(5*resampleNUM)] + words3_1[(4*resampleNUM):(5*resampleNUM)] + words4_1[(4*resampleNUM):(5*resampleNUM)] + words5_1[(4*resampleNUM):(5*resampleNUM)] + words6_1[(4*resampleNUM):(5*resampleNUM)] + words7_1[(4*resampleNUM):(5*resampleNUM)] + words8_1[(4*resampleNUM):(5*resampleNUM)] + words9_1[(4*resampleNUM):(5*resampleNUM)] + words10_1[(4*resampleNUM):(5*resampleNUM)] + \
                    words1_1[(5*resampleNUM):(6*resampleNUM)] + words2_1[(5*resampleNUM):(6*resampleNUM)] + words3_1[(5*resampleNUM):(6*resampleNUM)] + words4_1[(5*resampleNUM):(6*resampleNUM)] + words5_1[(5*resampleNUM):(6*resampleNUM)] + words6_1[(5*resampleNUM):(6*resampleNUM)] + words7_1[(5*resampleNUM):(6*resampleNUM)] + words8_1[(5*resampleNUM):(6*resampleNUM)] + words9_1[(5*resampleNUM):(6*resampleNUM)] + words10_1[(5*resampleNUM):(6*resampleNUM)] + \
                    words1_1[(6 * resampleNUM):(7 * resampleNUM)] + words2_1[(6 * resampleNUM):(7 * resampleNUM)] + words3_1[(6 * resampleNUM):(7 * resampleNUM)] + words4_1[(6 * resampleNUM):(7 * resampleNUM)] + words5_1[(6 * resampleNUM):(7 * resampleNUM)] + words6_1[(6 * resampleNUM):(7 * resampleNUM)] + words7_1[(6 * resampleNUM):(7 * resampleNUM)] + words8_1[(6 * resampleNUM):(7 * resampleNUM)] + words9_1[(6 * resampleNUM):(7 * resampleNUM)] + words10_1[(6 * resampleNUM):(7 * resampleNUM)] + \
                    words1_1[(7 * resampleNUM):end_pos7] + words2_1[(7 * resampleNUM):end_pos7] + words3_1[(7 * resampleNUM):end_pos7] + words4_1[(7 * resampleNUM):end_pos7] + words5_1[(7 * resampleNUM):end_pos7] + words6_1[(7 * resampleNUM):end_pos7] + words7_1[(7 * resampleNUM):end_pos7] + words8_1[(7 * resampleNUM):end_pos7] + words9_1[(7 * resampleNUM):end_pos7] + words10_1[(7 * resampleNUM):end_pos7] + \
                    words1_1[end_pos7:end_pos8] + words2_1[end_pos7:end_pos8] + words3_1[end_pos7:end_pos8] + words4_1[end_pos7:end_pos8] + words5_1[end_pos7:end_pos8] + words6_1[end_pos7:end_pos8] + words7_1[end_pos7:end_pos8] + words8_1[end_pos7:end_pos8] + words9_1[end_pos7:end_pos8] + words10_1[end_pos7:end_pos8] + \
                    words1_1[end_pos8:end_pos9] + words2_1[end_pos8:end_pos9] + words3_1[end_pos8:end_pos9] + words4_1[end_pos8:end_pos9] + words5_1[end_pos8:end_pos9] + words6_1[end_pos8:end_pos9] + words7_1[end_pos8:end_pos9] + words8_1[end_pos8:end_pos9] + words9_1[end_pos8:end_pos9] + words10_1[end_pos8:end_pos9]
            data = [float(c) if c!='nan' else np.nan for c in words[1:]]
            pcc_test = data[resampleNUM2:(2*resampleNUM2)]
            NMAE_test = data[(5*resampleNUM2):(6 * resampleNUM2)]

            fitting_interc = data[(6*resampleNUM2):(7 * resampleNUM2)]
            fitting_coefs = data[(7 * resampleNUM2):((7+coef_scaler_paramNUM) * resampleNUM2)]
            fitting_coefs = [fitting_coefs[group_i*coef_scaler_paramNUM:(group_i+1)*coef_scaler_paramNUM] for group_i in range(resampleNUM2)]
            scaler_means = data[((7 + coef_scaler_paramNUM) * resampleNUM2):((7 + 2 * coef_scaler_paramNUM) * resampleNUM2)]
            scaler_means = [scaler_means[group_i * coef_scaler_paramNUM:(group_i + 1) * coef_scaler_paramNUM] for group_i in range(resampleNUM2)]
            scaler_sds = data[((7 + 2 * coef_scaler_paramNUM) * resampleNUM2):((7 + 3 * coef_scaler_paramNUM) * resampleNUM2)]
            scaler_sds = [scaler_sds[group_i * coef_scaler_paramNUM:(group_i + 1) * coef_scaler_paramNUM] for group_i in range(resampleNUM2)]
            fitting_interc_good = fitting_interc
            fitting_coefs_good = fitting_coefs
            scaler_means_good = scaler_means
            scaler_sds_good = scaler_sds
            fitting_coefs_good = list(zip(*fitting_coefs_good))
            scaler_means_good = list(zip(*scaler_means_good))
            scaler_sds_good = list(zip(*scaler_sds_good))
            ### use mean values of params. as final model
            fitting_interc_mean = np.nanmean(fitting_interc_good)
            fitting_coefs_mean = [np.nanmean(c) for c in fitting_coefs_good]
            fitting_scalerMeans_mean = [np.nanmean(c) for c in scaler_means_good]
            fitting_scalerSds_mean = [np.nanmean(c) for c in scaler_sds_good]

            pcc_test = [c for c in pcc_test if not np.isnan(c)]
            NMAE_test = [c for c in NMAE_test if not np.isnan(c)]
            if not pcc_test:  # all are nan
                pcc_test = [0.01 for _ in range(resampleNUM2)]
            pcc_test_mean = np.nanmean(pcc_test)
            NMAE_test_mean = np.nanmean(NMAE_test)
            pcc_pval_test = stats.wilcoxon(np.array(pcc_test) - 0.3, alternative='greater')[1]
            NMAE_pval_test = stats.wilcoxon(np.array(NMAE_test) - 1, alternative='less')[1]
            pcc_all_raw[line_i] = pcc_test
            pcc_mean_list[line_i] = pcc_test_mean
            pcc_pval_list[line_i] = pcc_pval_test
            NMAE_mean_list[line_i] = NMAE_test_mean
            NMAE_pval_list[line_i] = NMAE_pval_test
            fitting_interc_mean_list[line_i] = fitting_interc_mean
            fitting_coefs_mean_list[line_i] = fitting_coefs_mean
            scaler_means_mean_list[line_i] = fitting_scalerMeans_mean
            scaler_sds_mean_list[line_i] = fitting_scalerSds_mean
        pcc_qval_list = list(multitest.multipletests(pcc_pval_list, method="fdr_bh")[1])
        NMAE_qval_list = list(multitest.multipletests(NMAE_pval_list, method="fdr_bh")[1])
        start_pos = ConF_ind * 6 + max(0, ConF_ind - 1) * 3
        result_list[1 + start_pos] = pcc_mean_list
        result_list[2 + start_pos] = pcc_pval_list
        result_list[3 + start_pos] = pcc_qval_list
        result_list[4 + start_pos] = NMAE_mean_list
        result_list[5 + start_pos] = NMAE_pval_list
        result_list[6 + start_pos] = NMAE_qval_list
        start_pos2 = ConF_ind*4
        param_list[1 + start_pos2] = fitting_interc_mean_list
        param_list[2 + start_pos2] = fitting_coefs_mean_list
        param_list[3 + start_pos2] = scaler_means_mean_list
        param_list[4 + start_pos2] = scaler_sds_mean_list

        if ConF_ind > 0:
            pcc_FC_list = list(np.array(pcc_mean_list)/pcc_mean_baseline_list)
            pcc_FC_pval_list = []
            for i_temp in range(geneNUM):
                try:
                    pcc_FC_pval_list.append(stats.median_test(pcc_all_raw[i_temp], pcc_raw_baseline_list[i_temp])[1])
                except:
                    pcc_FC_pval_list.append(1)
            pcc_FC_qval_list = list(multitest.multipletests(pcc_FC_pval_list, method="fdr_bh")[1])
            result_list[7 + start_pos] = pcc_FC_list
            result_list[8 + start_pos] = pcc_FC_pval_list
            result_list[9 + start_pos] = pcc_FC_qval_list
        else:
            pcc_mean_baseline_list = [max(c, 0.01) for c in pcc_mean_list]
            pcc_raw_baseline_list = pcc_all_raw
    confounder_best = ['' for _ in range(geneNUM)]
    pcc_best = ['' for _ in range(geneNUM)]
    pcc_pval_best = ['' for _ in range(geneNUM)]
    NMAE_best = ['' for _ in range(geneNUM)]
    NMAE_pval_best = ['' for _ in range(geneNUM)]

    for line_i in range(len(result_list[0])):
        if confounderNUM == 6:
            pcc_all_temp = [result_list[1][line_i], result_list[7][line_i], result_list[16][line_i], result_list[25][line_i],
                            result_list[34][line_i], result_list[43][line_i]] # change with conF number
        elif confounderNUM == 4:
            pcc_all_temp = [result_list[1][line_i], result_list[7][line_i], result_list[16][line_i],
                            result_list[25][line_i]]  # change with conF number
        elif confounderNUM == 3:
            pcc_all_temp = [result_list[1][line_i], result_list[7][line_i], result_list[16][line_i]]  # change with conF number
        else:
            raise Exception('confounderNUM=%d not supported!'%confounderNUM)
        max_ind = np.argmax(pcc_all_temp)
        if max_ind:
            compare_pval = result_list[8 + max_ind * 6 + max(0, max_ind - 1) * 3][line_i]
            if compare_pval < 0.05:
                confounder_best[line_i] = confounder_names[max_ind]
                pcc_best[line_i] = result_list[1 + max_ind * 6 + max(0, max_ind - 1) * 3][line_i]
                pcc_pval_best[line_i] = result_list[2 + max_ind * 6 + max(0, max_ind - 1) * 3][line_i]
                NMAE_best[line_i] = result_list[4 + max_ind * 6 + max(0, max_ind - 1) * 3][line_i]
                NMAE_pval_best[line_i] = result_list[5 + max_ind * 6 + max(0, max_ind - 1) * 3][line_i]
            else:
                max_ind = 0
        if not max_ind: # 'None' is the best confounder
            confounder_best[line_i] = confounder_names[max_ind]
            pcc_best[line_i] = result_list[1][line_i]
            pcc_pval_best[line_i] = result_list[2][line_i]
            NMAE_best[line_i] = result_list[4][line_i]
            NMAE_pval_best[line_i] = result_list[5][line_i]
        ############## Extract params. for fitting
        start_pos2 = max_ind * 4
        fitting_interc = param_list[1 + start_pos2][line_i] # single value
        fitting_coefs = param_list[2 + start_pos2][line_i] # list
        scaler_means = param_list[3 + start_pos2][line_i] # list
        scaler_sds = param_list[4 + start_pos2][line_i] # list

    pcc_qval_best = list(multitest.multipletests(pcc_pval_best, method="fdr_bh")[1])
    NMAE_qval_best = list(multitest.multipletests(NMAE_pval_best, method="fdr_bh")[1])
    result_list[-7] = confounder_best
    result_list[-6] = pcc_best
    result_list[-5] = pcc_pval_best
    result_list[-4] = pcc_qval_best
    result_list[-3] = NMAE_best
    result_list[-2] = NMAE_pval_best
    result_list[-1] = NMAE_qval_best

    predictable_gene_ind = [c for c in range(geneNUM) if ((pcc_best[c] > PCC_cutoff) and (pcc_qval_best[c] < FDR_cutoff) and (NMAE_best[c] < NMAE_cutoff) and (NMAE_qval_best[c] < FDR_cutoff))]
    geneNames_predictable = list([geneNames_all[c] for c in predictable_gene_ind])

    fnOut = open(fnOutNA, 'w')
    content_header0 = ['gene']
    content_header_none = ['pcc','pcc_pval','pcc_qval','NMAE','NMAE_pval','NMAE_qval']
    content_header_Alc = [c + '_Alc' for c in content_header_none] + ['pcc_FC_Alc', 'pcc_FC_pval_Alc', 'pcc_FC_qval_Alc']
    content_header_Sex = [c + '_Sex' for c in content_header_none] + ['pcc_FC_Sex', 'pcc_FC_pval_Sex', 'pcc_FC_qval_Sex']
    content_header_Age = [c + '_Age' for c in content_header_none] + ['pcc_FC_Age', 'pcc_FC_pval_Age', 'pcc_FC_qval_Age']
    content_header_HPV = [c + '_HPV' for c in content_header_none] + ['pcc_FC_HPV', 'pcc_FC_pval_HPV', 'pcc_FC_qval_HPV']
    content_header_Tob = [c + '_Tob' for c in content_header_none] + ['pcc_FC_Tob', 'pcc_FC_pval_Tob', 'pcc_FC_qval_Tob']
    content_header_best = ['confounder_best'] + [c+'_best' for c in content_header_none]
    content_all = ','.join(content_header0 + content_header_none + content_header_Alc + content_header_Sex +
                  content_header_Age + content_header_HPV + content_header_Tob +content_header_best)
    fnOut.write(content_all+'\n')

    for line_i in range(len(result_list[0])):
        content = [c[line_i] for c in result_list]
        content = ','.join([str(c) for c in content])
        fnOut.write(content + '\n')
    fnOut.close()

    fnOut3 = open(fnOutNA3, 'a')
    content = [cn, paired_Tumor_PBMC_dict[cn], 'Merge'] + geneNames_predictable
    panCell_mergedGeneLists[0] += geneNames_predictable
    fnOut3.write(','.join(content) + '\n')
    for ConF_i in range(confounderNUM):
        predictable_gene_ind_temp = [True if ((confounder_best[c] == confounder_names[ConF_i]) and (pcc_best[c] > PCC_cutoff) and (pcc_qval_best[c] < FDR_cutoff) and (NMAE_best[c] < NMAE_cutoff) and (NMAE_qval_best[c] < FDR_cutoff)) else False for c in range(geneNUM)]
        geneNames_predictable_temp = list(geneNames_all[predictable_gene_ind_temp])
        panCell_mergedGeneLists[ConF_i+1] += geneNames_predictable_temp
        content = [cn, paired_Tumor_PBMC_dict[cn], confounder_names[ConF_i]] + geneNames_predictable_temp
        fnOut3.write(','.join(content) + '\n')
geneNames_predictable = list(set(panCell_mergedGeneLists[0]))
content = ['All', 'Merge', 'Merge'] + geneNames_predictable
fnOut3.write(','.join(content) + '\n')
for ConF_i in range(confounderNUM):
    geneNames_predictable = list(set(panCell_mergedGeneLists[ConF_i+1]))
    content = ['All', 'Merge', confounder_names[ConF_i]] + geneNames_predictable
    fnOut3.write(','.join(content) + '\n')
fnOut3.close()

EndTime = time.time()
print('All done! Time used: %.2f'%(EndTime - StartTime))
