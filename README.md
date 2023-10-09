## How to reproduce the results

### Overview
There are a few main code files. "newModelsT5Simplified.py", "newModelsT5SimplifiedCpr.py", "newModelsT5SimplifiedRpl.py",  define the selective cache, compressive cache, and replacement-based selective cache respectively. The first one also includes the vanilla transformer and Xl cache. 

"cacheLMpg19.py", "cacheLMwiki.py", "cacheLMdog.py" control the experiments on PG-19, WikiText2, and CMU-DoG datasets.

### Dependencies

    torch
    transformers
    datasets
    evaluate
    apex

### Data
1. PG-19 
Download the PG-19 dataset from https://github.com/deepmind/pg19 First click "Full dataset download link", then select "train", "test", "validation" folders. Unzip the dataset into ./LMdatasets
The final "LMdatasets" folder should contain "train", "test", "validation" and "train_subset.txt". The last one is the list of the selected 1% data, it is already included in this repo.

2. WikiText2
The "datasets" library will automatically download WikiText2.

3. CMU-DoG
The concatenated version that used in the paper is already included in this repo at "./LMdatasets/merged-docs2"

4. Eye-tracking Corpora
The trained fixation prediction model is available in this repo at "./FPmodels/T5-tokenizer-BiLSTM-TRT-12-concat-3". There is no need to download eye-tracking corpora and train the fixation prediction model again. Nevertheless, the code is available, i.e., "trainFPLSTM.py" and "utilsFP.py"

### Run the experiments
Here are all the commands used to reproduce results in the main table. The first command corresponds to the first cell in the column, the second corresponds to the second cell, etc.

"--sec_cache_size" stands for secondary cache size.
"--cache slc/rpl/cpr" stands for selective, replacement-based selective, and compressive cache respectively.
We recommend running experiments with 1/2/4 GPUs

PG-19 Small (first column)

    python cacheLMpg19.py 
    python cacheLMpg19.py --xl_cache_size 512
    python cacheLMpg19.py --xl_cache_size 256 --sec_cache_size 256 --cache cpr
    python cacheLMpg19.py --xl_cache_size 256 --sec_cache_size 256 
    python cacheLMpg19.py --xl_cache_size 256 --sec_cache_size 256 --RSI
    python cacheLMpg19.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 5
    python cacheLMpg19.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 5 --RSI

    python cacheLMpg19.py --xl_cache_size 640
    python cacheLMpg19.py --xl_cache_size 384 --sec_cache_size 256 --cache cpr
    python cacheLMpg19.py --xl_cache_size 128 --sec_cache_size 512 --snippet_size 128 --snip_list_len 5 --cache rpl


WikiText2 Small (second column)

    python cacheLMwiki.py 
    python cacheLMwiki.py --xl_cache_size 512
    python cacheLMwiki.py --xl_cache_size 256 --sec_cache_size 256 --cache cpr
    python cacheLMwiki.py --xl_cache_size 256 --sec_cache_size 256 
    python cacheLMwiki.py --xl_cache_size 256 --sec_cache_size 256 --RSI
    python cacheLMwiki.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 3
    python cacheLMwiki.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 3 --RSI

    python cacheLMwiki.py --xl_cache_size 640
    python cacheLMwiki.py --xl_cache_size 384 --sec_cache_size 256 --cache cpr
    python cacheLMwiki.py --xl_cache_size 128 --sec_cache_size 512 --snippet_size 128 --snip_list_len 3 --cache rpl

WikiText2 Base (third column)

    add "--model base --batch_size 16  --acc_steps 2" to all commands above


CMU-DoG Small (fourth column)

    python cacheLMdog.py 
    python cacheLMdog.py --xl_cache_size 512
    python cacheLMdog.py --xl_cache_size 256 --sec_cache_size 256 --cache cpr
    python cacheLMdog.py --xl_cache_size 256 --sec_cache_size 256 
    python cacheLMdog.py --xl_cache_size 256 --sec_cache_size 256 --RSI
    python cacheLMdog.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 9
    python cacheLMdog.py --xl_cache_size 128 --sec_cache_size 384 --snippet_size 128 --snip_list_len 9 --RSI

    python cacheLMdog.py --xl_cache_size 640
    python cacheLMdog.py --xl_cache_size 384 --sec_cache_size 256 --cache cpr
    python cacheLMdog.py --xl_cache_size 128 --sec_cache_size 512 --snippet_size 128 --snip_list_len 9 --cache rpl

CMU-DoG Base (fifth column)

    add "--model base" to all commands above
