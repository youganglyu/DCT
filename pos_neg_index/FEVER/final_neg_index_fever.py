import numpy as np


for ttt in range(1,6):
    epoch = str(ttt)
    a=np.load('fever_all_neg_epoch'+epoch+'.npy',allow_pickle=True)
    b=a.item()
    index=[]
    for i in range(len(b)):
        index.append(b[i])
    index=np.array(index)
    #生成fever数据集的hard neg index
    np.save('fever_tbert_negindex_epoch'+epoch,index)



    hard_index = np.load('hard_index_fever.npy', allow_pickle=True)
    con_neg = list(index)
    for i in range(len(hard_index)):
        con_neg.append(con_neg[hard_index[i]])
    con_neg = np.array(con_neg)
    #生成fever hard数据集的hard neg index
    np.save('fever_hard_tbert_negindex_epoch'+epoch, con_neg)