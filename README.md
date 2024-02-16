# svp-sdml 

 - Data (Word-in-Context)
   - [AM2iCo](https://github.com/cambridgeltl/AM2iCo)
   - [MCL-WiC](https://github.com/SapienzaNLP/mcl-wic)
   - [XL-WiC](https://pilehvar.github.io/xlwic/)
 - Data (Semantic Change Detection)
   - [SemEval-2020 Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/)
   - [RuShiftEval](https://github.com/akutuzov/rushifteval_public)

# 1. Setup
## 1.1 Load environments
use Dockerfile and
```
setup.sh
```

## 1.2 Preprocess WiC datasets
we provide shell script of preprocessing
```
preprocess_wic.sh
```


### AM2iCo
```
python3 src/preprocess_wic.py \
    --am2ico_path path/to/am2ico/lang/data.tsv \
    --filename path/to/result/lang/data.tsv
```

### MCL-WiC
```
python3 src/preprocess_wic.py \
    --mclwic_data path/to/mclwic/lang/data.data \
    --mclwic_gold path/to/mclwic/lang/data.gold \
    --filename path/to/result/lang/data.tsv
```

### XL-WiC
```
python3 src/preprocess_wic.py \
    --xlwic_path path/to/xlwic/lang/data.txt \
    --filename path/to/result/lang/data.tsv
```

## 1.3 Calculate vectors
```
python3 src/prepare_vec_wic.py \
    --data_dir path/to/wic/processed/data/dir
```

# 2. Main
## 2.1 Load environments
use Dockerfile or
```
pip install -r requirements.txt
```


## 2.2 Training
```
python3 src/main.py \
    --train_data_pathes path/to/wic/vec/train \
    --dev_data_pathes path/to/wic/vec/dev \
    --test_data_pathes path/to/wic/vec/test \
    --search_param
```

## 2.3 Calculate vectors
SemEval
```
python3 src/prepare_vec_semeval.py \
    --file_path path/to/semeval/token.txt \
    --lemma_path path/to/semeval/lemma.txt \
    --target_words_list path/to/targets.txt \
    --lang [en, de, sw, la] \
    --output_name path/to/result/pkl
```

RuShiftEval
```
python3 src/prepare_vec_rushifteval.py \
    --file_path path/to/rushifteval/data.tsv \
    --target_words_list path/to/rushifteval/targets.txt \
    --output_name path/to/result/pkl
```

## 2.4 Evaluate
```
python3 src/evaluate.py \
    --gold_path path/to/graded.txt \
    --model_path path/to/model.pkl \
    --vec_pathes path/to/result/pkl
```
