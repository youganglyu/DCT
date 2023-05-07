import numpy as np

for ttt in range(1,6):
    epoch = str(ttt)
    emb=np.load('snli_tbert_emb_epoch'+epoch+'.npy')
    labels=np.load('snli_labels.npy')


    dic_01_all={}
    embs01=[]
    num01=0

    dic_02_all={}
    embs02=[]
    num02=0


    dic_12_all={}
    embs12=[]
    num12=0

    for i in range(len(labels)):
        if i%10000==0:
            print(i)
        if labels[i]==0:
            embs01.append(emb[i])
            dic_01_all[num01] = i
            num01 += 1

        if labels[i]==1:
            embs02.append(emb[i])
            dic_02_all[num02] = i
            num02 += 1

        if labels[i]==2:
            embs12.append(emb[i])
            dic_12_all[num12] = i
            num12 += 1
    print(num01)
    print(num02)
    print(num12)
    embs01=np.array(embs01).astype('float32')
    embs02=np.array(embs02).astype('float32')
    embs12=np.array(embs12).astype('float32')
    np.save('snli0_tbert_emb_epoch'+epoch,embs01)
    np.save('snli1_tbert_emb_epoch'+epoch, embs02)
    np.save('snli2_tbert_emb_epoch'+epoch, embs12)

    np.save('snli0_tbert_all', dic_01_all)
    np.save('snli1_tbert_all', dic_02_all)
    np.save('snli2_tbert_all', dic_12_all)
