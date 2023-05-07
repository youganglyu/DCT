import numpy as np
d = 128                           # 向量维度
nb = 100000                         # 向量集大小
nq = 10000                          # 查询次数
           # 随机种子,使结果可复现
for ttt in range(1,6):
    #search for 0 label

    epoch = str(ttt)
    xb=np.load('fever12_tbert_emb_hard_epoch'+epoch+'.npy').astype('float32')
    xb=xb.reshape(-1,128)
    print(xb.shape)

    xq =np.load('fever0_tbert_emb_epoch'+epoch+'.npy').astype('float32')
    xq=xq.reshape(-1,128)
    print(xq.shape)
    dict_load=np.load('fever12_tbert_all_hard.npy',allow_pickle=True)
    dict_load=dict_load.item()
    print(type(dict_load))

    dict0_load=np.load('fever0_tbert_all.npy',allow_pickle=True)
    dict0_load=dict0_load.item()


    import faiss
    nlist = 100
    k = 5
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    print(index.is_trained)
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    index.add(xb)                  # 添加索引可能会有一点慢
    D, I = index.search(xq, k)     # 搜索
    print(I[-5:])
    print(I[0])
    dic={}
    for i in range(I.shape[0]):
        temp=I[i]
        dic[dict0_load[i]]=dict_load[temp[0]]
    np.save('fever0_neg_epoch'+epoch,dic)
    #search for 1 label

    xb = np.load('fever02_tbert_emb_hard_epoch'+epoch+'.npy').astype('float32')
    xb = xb.reshape(-1, 128)
    print(xb.shape)

    xq = np.load('fever1_tbert_emb_epoch'+epoch+'.npy').astype('float32')
    xq = xq.reshape(-1, 128)
    print(xq.shape)
    dict_load = np.load('fever02_tbert_all_hard.npy', allow_pickle=True)
    dict_load = dict_load.item()
    print(type(dict_load))

    dict0_load = np.load('fever1_tbert_all.npy', allow_pickle=True)
    dict0_load = dict0_load.item()

    import faiss

    nlist = 100
    k = 5
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    print(index.is_trained)
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    index.add(xb)  # 添加索引可能会有一点慢
    D, I = index.search(xq, k)  # 搜索
    print(I[-5:])
    print(I[0])
    dic = {}
    for i in range(I.shape[0]):
        temp = I[i]
        dic[dict0_load[i]] = dict_load[temp[0]]
    np.save('fever1_neg_epoch'+epoch, dic)

    #搜索标签为2的

    xb = np.load('fever01_tbert_emb_hard_epoch'+epoch+'.npy').astype('float32')
    xb = xb.reshape(-1, 128)
    print(xb.shape)

    xq = np.load('fever2_tbert_emb_epoch'+epoch+'.npy').astype('float32')
    xq = xq.reshape(-1, 128)
    print(xq.shape)
    dict_load = np.load('fever01_tbert_all_hard.npy', allow_pickle=True)
    dict_load = dict_load.item()
    print(type(dict_load))

    dict0_load = np.load('fever2_tbert_all.npy', allow_pickle=True)
    dict0_load = dict0_load.item()

    import faiss

    nlist = 100
    k = 5
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    print(index.is_trained)
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    index.add(xb)  # 添加索引可能会有一点慢
    D, I = index.search(xq, k)  # 搜索
    print(I[-5:])
    print(I[0])
    dic = {}
    for i in range(I.shape[0]):
        temp = I[i]
        dic[dict0_load[i]] = dict_load[temp[0]]
    np.save('fever2_neg_epoch'+epoch, dic)