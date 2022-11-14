#### Usage: python 02_1.batchRun.py
########################################################################################################################
# Function: Run 02_2.geneExp_Predictor.py in batch
########################################################################################################################

import os
import time

resampleNUM = 100 # do bootstrapping training:test split for N times. In real case, 100 is used. As a demo, can set to 10
geneNUM = 40000 # process the first N genes. In real case, 40000 is used. As a demo, can set to 10

StartTime = time.time()

for randSeed in range(10): # 10 batches
    for cellType in range(12): # 12 cell types
        for conF in range(7): # 7 confounders
            command = 'python -W ignore 02_2.geneExp_Predictor.py'+' '+str(randSeed)+' '+str(resampleNUM)+' '+str(cellType)+' '+str(conF)+' '+str(geneNUM)
            print(command)
            os.system(command)
EndTime = time.time()
print('All done! Time used: %.2f'%(EndTime - StartTime))