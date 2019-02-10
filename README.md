# DMT

VarDial19 shared task: discriminating between mainland and taiwan variation of mandarin chinese -- [DMT](https://sites.google.com/view/vardial2019/campaign)

### Dataset

Download dmt data [here](https://www.dropbox.com/sh/dkhpcbyhmh89cdz/AAAu8L5j4Froml7reFQ1gQlfa?dl=0), put it in `raw_data` dir.

### Pre-processing
```
python3 preprocess.py
```

### Train
```
python3 train.py
```


### Performance

- Simplified

|  model                               | val_acc  | val_f1  |  train_time(one titan x) |
|--------------------------------------|----------|---------|--------------------------|
|simplified_bilstm_word_w2v_data_tune  |0.8855    | 0.8832  |00:08:26|
|simplified_dcnn_word_w2v_data_tune    |0.886     | 0.8846  |00:02:08|
|simplified_dpcnn_word_w2v_data_tune   |0.896     | 0.8952  |00:00:39|
|simplified_han_word_w2v_data_tune     |0.891     | 0.8931  |00:06:50|
|simplified_multicnn_word_w2v_data_tune|0.5       | 0.0     |00:01:13|
|simplified_rcnn_word_w2v_data_tune    |0.89      | 0.8879  |00:08:27|
|simplified_rnncnn_word_w2v_data_tune  |0.8925    | 0.8970  |00:06:09|
|simplified_cnn_word_w2v_data_tune     |0.893     | 0.8926  |00:00:43|
|simplified_vdcnn_word_w2v_data_tune   |0.8715    | 0.8704  |00:18:03|
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

