import sys
import copy
import torch
import numpy as np
import pandas as pd

# Output: Spare and Dense Features & Labels
# Input:  Files Path, Train-Valid-Test Ratio, Feature IDs, Perdiction Target, Window Size
def load_data_SD(File, TVT, FID, Target, W, device):
    # Initialization
    print('Reading File ...')
    DF = pd.read_csv(File).values.tolist()
    print('Finish Reading')
    XS, XD, Y, B = [], [], [], [0]      # XS & XD & Y & Batch ID
    NFID = copy.deepcopy(FID)           # New Feature ID (without label)
    NFID.pop(Target)
    NF = len(NFID)                      # Number of features
    pid, PC, rc = 0, 0, 0               # Patient ID, Patients Count, Record Count
    FFlag = 0                           # Feature Flage (Is Dense)
    SBucket, DBucket = [[0 for i in range(NF)] for j in range(W)], [[] for i in range(NF)]
    # Read Lines
    for D in DF:
        if pid != int(D[2]):            # New patient
            pid = int(D[2])
            PC += 1
            B.append(B[-1]+rc)
            rc, FFlag = 0, 0
            SBucket, DBucket = [[0 for i in range(NF)] for j in range(W)], [[] for i in range(NF)]
            print(PC, end='\r')
        fid, V = int(D[4]), D[9]        # Feature ID & Value
        if FID.index(fid)!=Target:      # Add features (not label)
            try:
                V, T = float(V), NFID.index(fid)
                if not np.isnan(V):
                    tmp = [0 for i in range(NF)]
                    tmp[T] = V
                    SBucket.append(tmp)
                    SBucket.pop(0)
                    DBucket[T] = DBucket[T][-(W-1):] + [V]
                    if FFlag==0:     # Update FFlag
                        if (np.asarray([len(DBucket[idf]) for idf in range(NF)])!=0).all(): FFlag = 1
            except: None
        if FID.index(fid)==Target and FFlag==1:                    # Add label (a new record)
            try:
                V, x = float(V), []
                if not np.isnan(V):
                    for idf in range(NF):
                        if len(DBucket[idf])<W: x.append([DBucket[idf][0] for i in range(W-len(DBucket[idf]))] + DBucket[idf] )
                        else:                   x.append( DBucket[idf][-W:])
                    XS.append(np.array(SBucket).T.tolist())
                    XD.append(x)
                    Y.append(V)
                    rc += 1
            except: None
    B.append(B[-1]+rc)
    B.pop(0)
    # Tensor
    BP0, BP1, BP2, BP3 = 0, int(len(B)*TVT[0]), int(len(B)*(TVT[0]+TVT[1])), len(B)-1
    XS_Tra, XD_Tra, Y_Tra, B_Tra = torch.FloatTensor(XS[B[BP0]:B[BP1]]).to(device), torch.FloatTensor(XD[B[BP0]:B[BP1]]).to(device), torch.FloatTensor(Y[B[BP0]:B[BP1]]).to(device), np.array(B[BP0:BP1+1])-B[BP0]
    XS_Val, XD_Val, Y_Val, B_Val = torch.FloatTensor(XS[B[BP1]:B[BP2]]).to(device), torch.FloatTensor(XD[B[BP1]:B[BP2]]).to(device), torch.FloatTensor(Y[B[BP1]:B[BP2]]).to(device), np.array(B[BP1:BP2+1])-B[BP1]
    XS_Tes, XD_Tes, Y_Tes, B_Tes = torch.FloatTensor(XS[B[BP2]:B[BP3]]).to(device), torch.FloatTensor(XD[B[BP2]:B[BP3]]).to(device), torch.FloatTensor(Y[B[BP2]:B[BP3]]).to(device), np.array(B[BP2:BP3+1])-B[BP2]
    print(np.shape(XS_Tra), np.shape(XS_Val), np.shape(XS_Tes))
    print(np.shape(XD_Tra), np.shape(XD_Val), np.shape(XD_Tes))
    print(np.shape(Y_Tra), np.shape(Y_Val), np.shape(Y_Tes))
    print(np.shape(B_Tra), np.shape(B_Val), np.shape(B_Tes))
    return XS_Tra,XD_Tra,Y_Tra,B_Tra, XS_Val,XD_Val,Y_Val,B_Val, XS_Tes,XD_Tes,Y_Tes,B_Tes