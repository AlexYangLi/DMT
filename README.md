# DMT

VarDial19 shared task: discriminating between mainland and taiwan variation of mandarin chinese -- [DMT](https://sites.google.com/view/vardial2019/campaign)

### Dataset

Download dmt data [here](https://www.dropbox.com/sh/dkhpcbyhmh89cdz/AAAu8L5j4Froml7reFQ1gQlfa?dl=0), put it in `raw_data` dir.

### Data augmentation

Cut long sentences into multiple shorter sub-sentences and replace the original sentence in the training data.

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
Conv + max pooling + Bidirectional gru + max pooling over time.

### ML based model

1. features  
binary ngram(1-3), tf ngram(1-3), tfidf ngram(1-3) in character & word level

2. models  
lr, svm, navie-bayers, random-forest, gradientboost, xgboost

### Dialect matching model

Learn similarities between the same dialects and dissimilarities between different dialects.

### Ensemble-based model

- mean ensemble

- max ensemble

- majortiy-vote ensemble


### Pre-processing
```
python3 preprocess.py
python3 prepare_ngram_fetures.py    # todo: merge into preprocess.py
python3 data_augment.py             # todo: merge into preprocess.py
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
|simplified_aug_bilstm_word_w2v_data_tune|0.8965  | 0.8965  |00:10:00|
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
4. data agumentaion doesn't help.

### Performance of ml based model

- Simplified
| model                                 | val_acc | val_f1 | val_p | val_r |
|---------------------------------------|---------|--------|-------|-------|
|simplified_svm_binary_char_(1, 1)      |0.81     |0.8087  |0.8144 |0.803  |
|simplified_svm_binary_char_(2, 2)      |0.8635   |0.8629  |0.8668 |0.859  |
|simplified_svm_binary_char_(3, 3)      |0.8705   |0.8687  |0.8808 |0.857  |
|simplified_svm_binary_char_(4, 4)      |0.853    |0.8498  |0.8684 |0.832  |
|simplified_svm_binary_char_(1, 3)      |0.8775   |0.8766  |0.8832 |0.87   |
|simplified_svm_binary_char_(2, 3)      |0.879    |0.8780  |0.8852 |0.871  |
|simplified_svm_binary_word_(1, 1)      |0.8375   |0.8344  |0.8504 |0.819  |
|simplified_svm_binary_word_(2, 2)      |0.6565   |0.7374  |0.5967 |0.965  |
|simplified_svm_binary_word_(3, 3)      |0.54     |0.6849  |0.5208 |1.0    |
|simplified_svm_binary_char_(2, 3)word_(1, 1) |0.877|0.8760|0.8831 |0.869  |
|simplified_svm_tf_char_(3, 3)          |0.8705   |0.8694  | 0.8769|0.862  |
|simplified_svm_tf_char_(2, 3)          |0.881    |0.8791  |0.8928 |0.866  |
|simplified_svm_tf_word_(1, 1)          |0.837    |0.8337  |0.8510 |0.817  |
|simplified_svm_tf_char_(2, 3)word_(1, 1) |0.88   |0.8787  |0.8877 |0.87   |
|simplified_svm_tfidf_char_(3, 3)       |  0.8935 |0.8926  |0.8994 |0.886  |
|simplified_svm_tfidf_char_(2, 3)       | 0.8955  |0.8946  |0.9023 |0.887  |
|simplified_svm_tfidf_word_(1, 1)       | 0.851   |0.8482  |0.8641 |0.833  |
|simplified_svm_tfidf_char_(2, 3)word_(1, 1) |0.8885|0.8881|0.8912 |0.885  |
|simplified_sgd_binary_char_(3, 3)      |0.8725   |0.8709  |0.8821 |0.86   |
|simplified_sgd_binary_char_(2, 3)      |0.872    |0.8698  |0.8851 |0.855  |
|simplified_sgd_binary_word_(1, 1)      |0.8455   |0.8446  |0.8493 |0.84   |
|simplified_sgd_binary_char_(2, 3)word_(1, 1) |0.8785|0.8759|0.8946|0.858  |
|simplified_sgd_tf_char_(3, 3)          |0.8655   |0.8582  |0.9075 |0.814  |
|simplified_sgd_tf_char_(2, 3)          |0.865    |0.8653  |0.8635 |0.867  |
|simplified_sgd_tf_word_(1, 1)          |0.8495   |0.8477  |0.8577 |0.838  |
|simplified_sgd_tf_char_(2, 3)word_(1, 1) |0.864  |0.8653  |0.8568 |0.874  |
|simplified_sgd_tfidf_char_(3, 3)       |0.876    |0.8745  |0.8852 |0.864  |
|simplified_sgd_tfidf_char_(2, 3)       |0.88     |0.8777  |0.8950 |0.861  |
|simplified_sgd_tfidf_word_(1, 1)       |0.853    |0.8469  |0.8837 |0.813  |
|simplified_sgd_tfidf_char_(2, 3)word_(1, 1) |0.8875|0.8866|0.8934 |0.88   |
|simplified_lr_binary_char_(1, 1)       |0.8185   |0.8167  |0.8246 |0.809  |
|simplified_lr_binary_char_(2, 2)       |0.872    |0.8701  |0.8835 |0.857  |
|simplified_lr_binary_char_(3, 3)       |0.879    |0.8775  |0.8883 |0.867  |
|simplified_lr_binary_char_(4, 4)       |0.8505   |0.8459  |0.8725 |0.821  |
|simplified_lr_binary_char_(2, 3)       |0.8865   |0.8854  |0.8939 |0.877  |
|simplified_lr_binary_char_(1, 3)       |0.886    |0.8847  |0.8947 |0.875  |
|simplified_lr_binary_word_(1, 1)       |0.859    |0.8545  |0.8827 |0.828  |
|simplified_lr_binary_word_(2, 2)       |0.706    |0.6111  |0.9023 |0.462  |
|simplified_lr_binary_word_(3, 3)       |0.552    |0.1899  |0.9905 |0.105  |
|simplified_lr_binary_char_(2, 3)word_(1, 1)  |0.8875|0.8865|0.8942|0.879  |
|simplified_mnb_binary_char_(1, 1)      |0.8225   |0.8222  |0.8235 |0.821  |
|simplified_mnb_binary_char_(2, 2)      |0.8935   |0.8942  |0.8885 |0.9    |
|simplified_mnb_binary_char_(3, 3)      |0.9015   |0.9035  |0.8859 |0.922  |
|simplified_aug_mnb_binary_char_(3, 3)  |0.8985   |0.9007  |0.8813 |0.921  |
|simplified_mnb_binary_char_(4, 4)      |0.8835   |0.8855  |0.8705 |0.901  |
|simplified_mnb_binary_char_(1, 3)      |0.903    |0.9040  |0.8951 |0.913  |
|simplified_mnb_binary_char_(2, 3)      |0.908    |0.9094  |0.8953 |0.924  |
|simplified_aug_mnb_binary_char_(2, 3)  |0.91     |0.9111  |0.8996 |0.923  |
|simplified_mnb_binary_word_(1, 1)      |0.878    |0.8790  |0.8720 |0.886  |
|simplified_mnb_binary_word_(2, 2)      |0.709    |0.6141  |0.9114 |0.463  |
|simplified_mnb_binary_word_(3, 3)      |0.552    |0.1898  |0.9905 |0.105  |
|simplified_aug_mnb_binary_word_(1, 1)  |0.872    |0.8756  |0.8516 |0.901  |
|simplified_mnb_binary_char_(2, 3)word_(1, 1) |0.9055|0.9070|0.8925|0.922  |
|simplified_aug_mnb_binary_char_(2, 3)word_(1, 1)|0.906|0.9076|0.8926|0.923|
|simplified_mnb_tf_char_(3, 3)          |0.901    |0.9030  |0.8848 |0.922  |
|simplified_mnb_tf_char_(2, 3)          |0.906    |0.9077  |0.8919 |0.924  |
|simplified_mnb_tf_word_(1, 1)          |0.8795   |0.8800  |0.8761 |0.884  |
|simplified_mnb_tf_char_(2, 3)word_(1, 1) |0.9035 |0.9052  |0.8898 |0.921  |
|simplified_mnb_tfidf_char_(3, 3)       |0.8945   |0.8969  |0.8768 |0.918  |
|simplified_mnb_tfidf_char_(2, 3)       |0.8995   |0.9011  |0.8867 |0.916  |
|simplified_mnb_tfidf_word_(1, 1)       |0.873    |0.8745  |0.8643 |0.885  |
|simplified_mnb_tfidf_char_(2, 3)word_(1, 1)|0.895|0.8970 |0.8805 |0.914   |



- traditional

| model                                 | val_acc | val_f1 | val_p | val_r |
|---------------------------------------|---------|--------|-------|-------|
|traditional_svm_binary_char_(1, 1)     |0.8325   |0.8321  |0.8342 |0.83   |
|traditional_svm_binary_char_(2, 2)     |0.8765   |0.8747  |0.8877 |0.862  |
|traditional_svm_binary_char_(3, 3)     |0.883    |0.8821  |0.8892 |0.875  |
|traditional_svm_binary_char_(4, 4)     |0.86     |0.8574  |0.8734 |0.842  |
|traditional_svm_binary_char_(1, 3)     |0.8895   |0.8891  |0.8922 |0.886  |
|traditional_svm_binary_word_(1, 1)     |0.8435   |0.8412  |0.8538 |0.829  |
|traditional_svm_binary_word_(2, 2)     |0.657    |0.7380  |0.5970 |0.966  |
|traditional_svm_binary_word_(3, 3)     |0.54     |0.6850  |0.5208 |1.0    |
|traditional_svm_binary_char_(2, 3)word_(1, 1)|0.897|0.8963|0.9026|0.89|
|traditional_lr_binary_char_(1, 1)      |0.8455   |0.8443  |0.8508 |0.838  |
|traditional_lr_binary_char_(2, 2)      |0.889    |0.8874  |0.9002 |0.875  |
|traditional_lr_binary_char_(3, 3       |0.884    |0.8825  |0.8943 |0.871  |
|traditional_lr_binary_char_(4, 4)      |0.857    |0.8529  |0.8782 |0.829  |
|traditional_lr_binary_char_(2, 3)      |0.896    |0.8947  |0.9057 |0.884  |
|traditional_lr_binary_char_(1, 3)      |0.899    |0.8979  |0.9080 |0.888  |
|traditional_lr_binary_word_(1, 1)      |0.8635   |0.8591  |0.8879 |0.832  |
|traditional_lr_binary_word_(2, 2)      |0.707    |0.6124  |0.9043 |0.463  |
|traditional_lr_binary_word_(3, 3)      |0.552    |0.1899  |0.9906 |0.105  |
|traditional_lr_binary_char_(2, 3)word_(1, 1)|0.899|0.8979|0.9071|0.889|
|traditional_mnb_binary_char_(1, 1)     |0.848    |0.8486  |0.8452 |0.852  |
|traditional_mnb_binary_char_(2, 2)     |0.91     |0.9104  |0.9059 |0.915  |
|traditional_mnb_binary_char_(3, 3)     |0.915    |0.9166  |0.8998 |0.934  |
|traditional_mnb_binary_char_(4, 4)     |0.891    |0.8930  |0.8767 |0.91   |
|traditional_aug_mnb_binary_char_(3, 3) |0.9105   |0.9122  |0.8951 |0.93   |
|traditional_mnb_binary_char_(2, 3)     |0.9225   |0.9234  |0.9130 |0.934  |
|traditional_mnb_binary_char_(1, 3)     |0.917    |0.9176  |0.9112 |0.924  |
|traditional_aug_mnb_binary_char_(2, 3) |0.923    |0.9237  |0.9155 |0.932  |
|traditional_mnb_binary_word_(1, 1)     |0.8855   |0.8864  |0.8791 |0.894  |
|traditional_aug_mnb_binary_word_(1, 1) |0.8795   |0.8829  |0.8584 |0.909  |
|traditional_mnb_binary_word_(2, 2)     |0.71     |0.6154  |0.9134 |0.464  |
|traditional_mnb_binary_word_(3, 3)     |0.552    |0.1899  |0.9905 |0.105  |
|traditional_mnb_binary_char_(2, 3)word_(1, 1)|0.92|0.9211|0.9085|0.934|
|traditional_aug_mnb_binary_char_(2, 3)word_(1, 1)|0.918|0.9192|0.9058|0.933|

- conclusion
1. char trigram is better than char unigram and char bigram, word unigram is better than word bigram and word trigram
2. combine bigram and trigram helps, but further combine word unigram doesn't make a difference
3. binary vectors, tf weighted vectors and tf-idf weighted vectors have very close performace
4. navie bayers is a very strong classifier on this task
5. data agumentation doesn't make a difference

### Performance of dialect matching model
Not helping.

### Performance of Ensemble-based model

- simplified

| ensemble_model                    | ensemble_type | val_acc | val_f1  | val_p  | val_r  |
|-----------------------------------|---------------|---------|---------|--------|--------|
|bilstm_word mnb_binary_char_(2, 3) |  mean         | 0.913   | 0.9137  | 0.9065 | 0.921  |
|bilstm_word mnb_ianry_char_(2, 3)  |  max          | 0.902   | 0.9064  | 0.8675 | 0.949  |
|bilstm_word mnb_ianry_char_(2, 3)  |  vote         | 0.9     | 0.8996  | 0.9032 | 0.896  |
|rcnn_word mnb_binary_char_(2, 3)   |  mean         | 0.914   | 0.9145  | 0.9065 | 0.921  |
|rcnn_word mnb_ianry_char_(2, 3)    |  max          | 0.904   | 0.9080  | 0.8713 | 0.948  |
|rcnn_word mnb_ianry_char_(2, 3)    |  vote         | 0.904   | 0.9080  | 0.8713 | 0.948  |

- traditional

| ensemble_model                    | ensemble_type | val_acc | val_f1  | val_p  | val_r  |
|-----------------------------------|---------------|---------|---------|--------|--------|
|bilstm_word mnb_binary_char_(2, 3) |  mean         | 0.924   | 0.9242  | 0.9223 | 0.926  |
|bilstm_word mnb_ianry_char_(2, 3)  |  max          | 0.9185  | 0.9209  | 0.8944 | 0.949  |
|bilstm_word mnb_ianry_char_(2, 3)  |  vote         | 0.9115  | 0.9097  | 0.9282 | 0.892  |
|rcnn_word mnb_binary_char_(2, 3)   |  mean         | 0.923   | 0.9233  | 0.9196 | 0.927  |
|rcnn_word mnb_ianry_char_(2, 3)    |  max          | 0.9165  | 0.9194  | 0.8888 | 0.952  |
|rcnn_word mnb_ianry_char_(2, 3)    |  vote         | 0.9165  | 0.9194  | 0.8889 | 0.952  |