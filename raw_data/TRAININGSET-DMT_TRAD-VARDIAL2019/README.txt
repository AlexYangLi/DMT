=== DMT ===

We present a task on discriminating between Mainland and Taiwan variation of Mandarin. For technical reasons the task will be presented in two tracks: traditional and simplified version of the data, where training and development test set present the same text in two different scripts, the test set contains the same sentences with different order of sentences for the both scripts.
This version presents texts in TRADITIONAL script.   

=== Data Format ===

Each line in the *.txt files is tab-delimited in the format:

    sentence<tab>variation-label

where variation-label is either M (for Mainland) or T (for Taiwan) variation. *.labels contain the second field of *.txt files, or only labels.

The training data contains the following files:
    train.txt - training set, 18770 sentences
    train.labels - labels only, 18770 labels, equally 9385 for each T and M labels

    dev.txt   - development/validation set, 2000 sentences, 1000 for each variation
    dev.labels - labels for the dev set, 1000 for each T and M labels


=== Data description ===


Like English, Mandarin has several varieties (standards) among the speaking communities. This task aims at discriminating between two major varieties of Mandarin Chinese: Putonghua (Mainland China) and Guoyu (Taiwan). The two variations of Mandarin are very close to each other and the main differences lie mostly in pronunciation. 

We provide a training set of 18770 sentences belonging to the domain of news for each of the Mandarin variation (9385 for each variation). For both variations, the sentences are tokenized and punctuation is removed from the texts. 
The main task will be to determine if a sentence belongs to news articles from Mainland China or from Taiwan. The scripts of the two variations are different: Mainland uses simplified while in Taiwan traditional characters are used. In order to unify the texts, we prepared two tracks, both for traditional and simplified.  

Conversion Simplified to Traditional; Traditional to Simplified characters was made by ‘opencc’ tool (in effect, coding sets with some lexical conversion as well). However, conversion cannot be 100% accurate in either direction, it will have some distortion/information lost.  Hence it is likely that a system may have different performance on the two converted versions, and that is why we introduced the same data under the two tracks.


=== Evaluation ===

The test data will be released later, they will contain the same amount of sentences as the development set (2000 sentences, 1000 for each variation), the respective labels will be not be provided. Sentences in the test set in the both ‘traditional’ and ‘simplified’ tracks belong to the same subset of the corpora, but the order in the two tracks is different.

Participants will be required to submit the labels for these test instances in both tracks.

The exact details of the submission file format will be provided later.
