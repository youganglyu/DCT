import numpy as np
for ttt in range(1,6):
    epoch = str(ttt)
    dict_load0=np.load('snli0_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load0=dict_load0.item()

    dict_load1=np.load('snli1_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load1=dict_load1.item()

    dict_load2=np.load('snli2_neg_epoch'+epoch+'.npy',allow_pickle=True)
    dict_load2=dict_load2.item()

    dict_all={}
    dict_all.update(dict_load0)
    dict_all.update(dict_load1)
    dict_all.update(dict_load2)
    np.save('snli_all_neg_epoch'+epoch,dict_all)
