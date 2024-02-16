echo Setup sentence encoder ...

git clone https://github.com/pierluigic/xl-lexeme.git

cd xl-lexeme

pip install .

cd ..


echo Download WiC datasets ...

mkdir data

cd data

git clone https://github.com/SapienzaNLP/mcl-wic.git

git clone https://github.com/cambridgeltl/AM2iCo.git

wget https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip



echo Download SCD datasets ...

wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip

wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip

wget https://zenodo.org/records/3992738/files/semeval2020_ulscd_lat.zip

wget https://zenodo.org/records/3730550/files/semeval2020_ulscd_swe.zip

git clone https://github.com/akutuzov/rushifteval_public.git