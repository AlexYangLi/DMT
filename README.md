# DMT

VarDial19 shared task: discriminating between mainland and taiwan variation of mandarin chinese -- [DMT](https://sites.google.com/view/vardial2019/campaign)

### Dataset

Download dmt data [here](https://www.dropbox.com/sh/dkhpcbyhmh89cdz/AAAu8L5j4Froml7reFQ1gQlfa?dl=0), put it in `raw_data` dir.

### DL based Models

- [TextCNN, EMNLP2014](https://www.aclweb.org/anthology/D14-1181)
Kim et al. Convolutional Neural Networks for Sentence Classification.

- [DCNN, ACL2014](http://www.aclweb.org/anthology/P14-1062)
Kalchbrenner et al. A Convolutional Neural Network for Modelling Sentences

- [RCNN, AAAI2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
Lai et al. Recurrent Convolutional Neural Networks for Text Classification.

- [HAN, NAACL-HLT2016](http://www.aclweb.org/anthology/N16-1174)
Yang et al. Hierarchical Attention Networks for Document Classification.

- [DPCNN, ACL2017](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)
Johnson et al. Deep Pyramid Convolutional Neural Networks for Text Categorization.

- [VDCNN, EACL2017](http://www.aclweb.org/anthology/E17-1104)
Conneau et al. Very Deep Convolutional Networks for Text Classification.

- MultiTextCNN
Extension of textcnn, stacking multiple cnns with the same filter size.

- BiLSTM
Bidirectional lstm + max pooling over time.

- RNNCNN
Bidirectional gru + conv + max pooling & avg pooling.

- CNNRNN
conv + max pooling + Bidirectional gru + max pooling over time.

### ML based model

1. features
binary ngram(1-3), tf ngram(1-3), tfidf ngram(1-3) in character & word level

2. models
lr, svm, navie-bayers, random-forest, gradientboost, xgboost

### Siamese model


### Pre-processing
```
python3 preprocess.py
python3 prepare_ngram_fetures.py
```

### Train
```
python3 train.py
```


### Performance of dl based models

- Simplified

|  model                               | val_acc  | val_f1  |  train_time(one titan x) |
|--------------------------------------|----------|---------|--------------------------|
|simplified_bilstm_word_w2v_data_tune  |0.9       | 0.8996  |00:08:26|
|simlifiedd_cnnrnn_word_w2v_data_tune  |0.8935    | 0.8923  |00:09:07|
|simplified_dcnn_word_w2v_data_tune    |0.897     | 0.8971  |00:02:08|
|simplified_dpcnn_word_w2v_data_tune   |0.8925    | 0.8932  |00:00:39|
|simplified_han_word_w2v_data_tune     |0.8915    | 0.8896  |00:06:50|
|simplified_multicnn_word_w2v_data_tune|0.5       | 0.0     |00:01:13|
|simplified_rcnn_word_w2v_data_tune    |0.8985    | 0.8965  |00:08:27|
|simplified_rnncnn_word_w2v_data_tune  |0.895     | 0.8960  |00:06:09|
|simplified_cnn_word_w2v_data_tune     |0.8965    | 0.8998  |00:00:43|
|simplified_vdcnn_word_w2v_data_tune   |0.871     | 0.8716  |00:18:03|
|simplified_bilstm_word_w2v_data_fix   |0.8445    | 0.8449  |00:16:23|
|simplified_dcnn_word_w2v_data_fix     |0.816     | 0.8122  |00:03:46|
|simplified_dpcnn_word_w2v_data_fix    |0.8245    | 0.8152  |00:00:57|
|simplified_han_word_w2v_data_fix      |0.8345    | 0.8346  |00:15:44|
|simplified_multicnn_word_w2v_data_fix |0.5       | 0.6667  |00:01:10|
|simplified_rcnn_word_w2v_data_fix     |0.844     | 0.8421  |00:12:03|
|simplified_rnncnn_word_w2v_data_fix   |0.8335    | 0.8316  |00:09:07|
|simplified_cnn_word_w2v_data_fix      |0.825     | 0.8227  |00:00:46|
|simplified_vdcnn_word_w2v_data_fix    |0.7935    | 0.7800  |00:11:59|
|simplified_bilstm_char_w2v_data_tune  |0.849     | 0.8442  |00:26:15|
|simplified_dcnn_char_w2v_data_tune    |0.8525    | 0.8509  |00:04:22|
|simplified_dpcnn_char_w2v_data_tune   |0.8605    | 0.8600  |00:01:06|
|simplified_han_char_w2v_data_tune     |0.5       | 0.0     |00:11:57|
|simplified_multicnn_char_w2v_data_tune|0.5       | 0.0     |00:02:17|
|simplified_rcnn_char_w2v_data_tune    |0.85      | 0.8416  |00:20:21|
|simplified_rnncnn_char_w2v_data_tune  |0.8585    | 0.8573  |00:16:44|
|simplified_cnn_char_w2v_data_tune     |0.874     | 0.8746  |00:02:41|
|simplified_vdcnn_char_w2v_data_tune   |0.5095    | 0.1264  |00:06:58|
|simplified_bilstm_char_w2v_data_fix   |0.809     | 0.8068  |00:35:03|
|simplified_dcnn_char_w2v_data_fix     |0.7565    | 0.7514  |00:04:49|
|simplified_dpcnn_char_w2v_data_fix    |0.7985    | 0.8000  |00:02:36|
|simplified_han_char_w2v_data_fix      |0.576     | 0.6428  |00:11:57|
|simplified_multicnn_char_w2v_data_fix |0.5       | 0.6667  |00:02:11|
|simplified_rcnn_char_w2v_data_fix     |0.8145    | 0.8179  |00:36:16|
|simplified_rnncnn_char_w2v_data_fix   |0.7945    | 0.7876  |00:19:18|
|simplified_cnn_char_w2v_data_fix      |0.8155    | 0.8073  |00:02:30|
|simplified_vdcnn_char_w2v_data_fix    |0.529     | 0.1751  |00:12:29|

- traditional

|  model                                | val_acc  | val_f1  |  train_time(one titan x) |
|---------------------------------------|----------|---------|--------------------------|
|traditional_bilstm_word_w2v_data_tune  |0.9115    | 0.9097  |00:09:07|
|traditional_cnnrnn_word_w2v_data_tune  |0.908     | 0.9079  |00:07:07|
|traditional_dcnn_word_w2v_data_tune    |0.908     | 0.9089  |00:02:17|
|traditional_dpcnn_word_w2v_data_tune   |0.907     | 0.9067  |00:01:06|
|traditional_han_word_w2v_data_tune     |0.902     | -       |00:07:18|
|traditional_multicnn_word_w2v_data_tune|0.5       | 0.6667  |00:01:30|
|traditional_rcnn_word_w2v_data_tune    |0.912     | 0.9110  |00:07:56|
|traditional_rnncnn_word_w2v_data_tune  |0.9095    | -       |00:06:38|
|traditional_cnn_word_w2v_data_tune     |0.909     | 0.9084  |00:01:00|
|traditional_vdcnn_word_w2v_data_tune   |0.885     | 0.8864  |00:13:08|
|traditional_bilstm_word_w2v_data_fix   |0.8475    | 0.8508  |00:15:56|
|traditional_dcnn_word_w2v_data_fix     |0.834     | 0.8302  |00:02:34|
|traditional_dpcnn_word_w2v_data_fix    |0.839     | 0.8382  |00:01:02|
|traditional_han_word_w2v_data_fix      |0.8535    | 0.8539  |00:11:47|
|traditional_multicnn_word_w2v_data_fix |0.5       | 0.6667  |00:01:26|
|traditional_rcnn_word_w2v_data_fix     |0.8545    | 0.8522  |00:12:38|
|traditional_rnncnn_word_w2v_data_fix   |0.8505    | 0.8510  |00:10:31|
|traditional_cnn_word_w2v_data_fix      |0.842     | 0.8326  |00:01:13|
|traditional_vdcnn_word_w2v_data_fix    |0.8005    | 0.7944  |00:11:22|
|traditional_bilstm_char_w2v_data_tune  |0.8635    | 0.8616  |00:24:57|
|traditional_dcnn_char_w2v_data_tune    |0.8705    | 0.8684  |00:04:57|
|traditional_dpcnn_char_w2v_data_tune   |0.884     | 0.8850  |00:02:59|
|traditional_han_char_w2v_data_tune     |0.5       | 0.0     |00:13:05|
|traditional_multicnn_char_w2v_data_tune|0.5       | 0.0     |00:02:32|
|traditional_rcnn_char_w2v_data_tune    |0.866     | 0.8640  |00:19:07|
|traditional_rnncnn_char_w2v_data_tune  |0.8665    | 0.8607  |00:19:32|
|traditional_cnn_char_w2v_data_tune     |0.878     | 0.8803  |00:02:41|
|traditional_vdcnn_char_w2v_data_tune   |0.5015    | 0.0404  |00:08:50|
|traditional_bilstm_char_w2v_data_fix   |0.825     | 0.8234  |00:44:19|
|traditional_dcnn_char_w2v_data_fix     |0.781     | 0.768   |00:06:27|
|traditional_dpcnn_char_w2v_data_fix    |0.8055    | 0.8052  |00:02:27|
|traditional_han_char_w2v_data_fix      |0.5       | 0.0     |00:12:30|
|traditional_multicnn_char_w2v_data_fix |0.5       | 0.00    |00:02:28|
|traditional_rcnn_char_w2v_data_fix     |0.8145    | 0.8179  |00:33:37|
|traditional_rnncnn_char_w2v_data_fix   |0.8025    | 0.8045  |00:27:34|
|traditional_cnn_char_w2v_data_fix      |0.823     | 0.8281  |00:03:52|
|traditional_vdcnn_char_w2v_data_fix    |0.5       | 0.0176  |00:07:40|

- conclusion
1. word level input is better than charracter level input
2. fine-tuning word embeddings performs better than fixing word embeddings
3. `BiLSTM` performs best, but other models except `vdccn` and `multicnn` performs very close.