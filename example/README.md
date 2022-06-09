# Walk-through Example

This file shows the rough procedure training an end-to-end SLT Transformer model based on MUST_C dataset.

### Step 1. Download MUST_C Dataset

Take the En->Ge as an example

* You can go to [the official website](https://ict.fbk.eu/must-c/) to download the dataset. 

* You can use the following 
[Google Drive Address](https://drive.google.com/open?id=1Mf2il_VelDIJMSio0bq7I8M9fSs-X4Ie) for downloading.


Untar the dataset.

### Step 2. Download this code base

```
git clone https://github.com/bzhangGo/st_from_scratch.git
```
Suppose the downloaded code path is `st_from_scratch` so we refer to the code base as `${code}`

### Step 3. Preprocess the speech dataset

You need to preprocess the English and German text file (tokenization, truecase, subword-bpe).
Audios will be dynamically loaded during training.

1) Preprocessing the text files
```
en_de=/path/to/untared/en-de/
ln -s ${en_de} en-de
ln -s en-de/data/dev/txt/dev.en .
ln -s en-de/data/dev/txt/dev.de .
ln -s en-de/data/tst-COMMON/txt/tst-COMMON.en test.en
ln -s en-de/data/tst-COMMON/txt/tst-COMMON.de test.de
ln -s en-de/data/train/txt/train.en .
ln -s en-de/data/train/txt/train.de .

# tokenize, true-case and BPE
# you need download the mosesdecoder and subword-nmt, and re-set the path in the following script 
./prepare.sh

# prepare vocabulary
python ${code}/vocab.py train.bpe.en vocab.zero.en
python ${code}/vocab.py train.bpe.de vocab.zero.de
```
The resulting file is: 
    
    - (train source, train target): `train.bpe.en, train.bpe.de`
    - (dev source, dev target): `dev.bpe.en, dev.bpe.de`
    - (test source, test target): `test.bpe.en, test.reftok.de`

Notice the test reference file: `test.reftok.de`. It's only tokenized, without punctuation normalizing and true-casing

### Step 4. Train your model

See the given running scripts `train.sh` for reference. It uses about 4~5 days (with one GPU) or shorter (with more gpus).

### Step 5. Decoding

See the given running scripts `test.sh` for reference.

