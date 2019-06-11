This is the specification file for conditional seq2seq VAE training and validation dataset - English tense

1. train.txt
The file is for training. There are 1227 training pairs.
Each training pair includes 4 words: simple present(sp), third person(tp), present progressive(pg), simple past(p).
sp tp pg p
0 1 2 3
2. test.txt
The file is for validating. There are 10 validating pairs.
Each training pair includes 2 words with different combination of tenses.
You have to follow those tenses to test your model.

Here are to details of the file:

sp -> p 0 3
sp -> pg 0 2
sp -> tp 0 1
sp -> tp 0 1
p  -> tp 3 1
sp -> pg 0 2
p  -> sp 3 0
pg -> sp 2 0
pg -> p 2 3
pg -> tp 2 1