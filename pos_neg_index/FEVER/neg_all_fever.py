import numpy as np
for ttt in range(1,6):
    epoch = str(ttt)
    dict_load0=np.load('fever0_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load0=dict_load0.item()

    dict_load1=np.load('fever1_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load1=dict_load1.item()

    dict_load2=np.load('fever2_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load2=dict_load2.item()

    dict_all={}
    dict_all.update(dict_load0)
    dict_all.update(dict_load1)
    dict_all.update(dict_load2)
    np.save('fever_all_neg_epoch'+epoch,dict_all)
