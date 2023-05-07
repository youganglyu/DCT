import numpy as np
preds_soft=np.load('mnli_tbert_bias_epoch5_soft3.npy',allow_pickle=True)
labels=np.load('mnli_labels.npy',allow_pickle=True)
preds_soft_max=np.max(preds_soft, axis=1)
print(preds_soft_max.shape)
print(labels)
labels_lol=labels
# print(labels)
hard_index=[]
print(len(labels_lol))
print(preds_soft.shape)
num=0
for i in range(len(labels_lol)):
    if preds_soft[i][labels_lol[i]]!=preds_soft_max[i] and preds_soft_max[i]>=0.8:
        hard_index.append(i)
    if preds_soft[i][labels_lol[i]]==preds_soft_max[i]:
        num+=1
print(float(num)/float(len(labels_lol)))
hard_index=np.array(hard_index)
print(hard_index.shape)
np.save('hard_index_mnli_8',hard_index)

