import numpy as np
for ttt in range(1,6):
    epoch = str(ttt)
    emb=np.load('mnli_tbert_emb_epoch'+epoch+'.npy')
    labels=np.load('mnli_labels.npy')


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
        if labels[i]==0 or labels[i]==1:
            embs01.append(emb[i])
            dic_01_all[num01] = i
            num01 += 1

        if labels[i]==0 or labels[i]==2:
            embs02.append(emb[i])
            dic_02_all[num02] = i
            num02 += 1

        if labels[i]==1 or labels[i]==2:
            embs12.append(emb[i])
            dic_12_all[num12] = i
            num12 += 1


    print(num01)
    print(num02)
    print(num12)
    embs01=np.array(embs01).astype('float32')
    embs02=np.array(embs02).astype('float32')
    embs12=np.array(embs12).astype('float32')
    np.save('mnli01_tbert_emb_epoch'+epoch, embs01)
    np.save('mnli02_tbert_emb_epoch'+epoch, embs02)
    np.save('mnli12_tbert_emb_epoch'+epoch, embs12)

    np.save('mnli01_tbert_all', dic_01_all)
    np.save('mnli02_tbert_all', dic_02_all)
    np.save('mnli12_tbert_all', dic_12_all)
