import numpy as np
labels=np.load('fever_labels.npy')
bias_data_index=np.load('hard_index_fever.npy')
for ttt in range(1,6):
    epoch = str(ttt)
    emb=np.load('fever_tbert_emb_epoch'+epoch+'.npy')
    dic_01_all={}
    embs01=[]
    num01=0

    dic_02_all={}
    embs02=[]
    num02=0


    dic_12_all={}
    embs12=[]
    num12=0
    for i in range(len(bias_data_index)):
        if labels[bias_data_index[i]]==0 or labels[bias_data_index[i]]==1:
            embs01.append(emb[bias_data_index[i]])
            dic_01_all[num01] = bias_data_index[i]
            num01 += 1
        if labels[bias_data_index[i]]==0 or labels[bias_data_index[i]]==2:
            embs02.append(emb[bias_data_index[i]])
            dic_02_all[num02] = bias_data_index[i]
            num02 += 1
        if labels[bias_data_index[i]]==1 or labels[bias_data_index[i]]==2:
            embs12.append(emb[bias_data_index[i]])
            dic_12_all[num12] = bias_data_index[i]
            num12 += 1

    print(num01)
    print(num02)
    print(num12)
    embs01=np.array(embs01).astype('float32')
    embs02=np.array(embs02).astype('float32')
    embs12=np.array(embs12).astype('float32')
    np.save('fever01_tbert_emb_hard_epoch'+epoch, embs01)
    np.save('fever02_tbert_emb_hard_epoch'+epoch, embs02)
    np.save('fever12_tbert_emb_hard_epoch'+epoch, embs12)

    np.save('fever01_tbert_all_hard', dic_01_all)
    np.save('fever02_tbert_all_hard', dic_02_all)
    np.save('fever12_tbert_all_hard', dic_12_all)

