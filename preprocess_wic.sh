cd /work/src

data_dir="/work/data/WiC"
result_dir="${data_dir}/processed"

mkdir "${result_dir}"

am2ico_dir="${data_dir}/AM2iCo/data"
xlwic_dir="${data_dir}/xlwic_datasets"
mclwic_dir="${data_dir}/MCL-WiC"

types="train dev test"

#AM2iCo
am2ico_langs="ar bn de eu fi id ja ka kk ko ru tr ur zh"
for lang in ${am2ico_langs};
do
	for type in ${types};
	do
		python3 preprocess_wic.py \
			--am2ico_path "${am2ico_dir}/${lang}/${type}.tsv" \
			--filename "am2ico.${type}.${lang}-en"
	done
done

mv preprocess_wic_debug.log "${result_dir}/"
mv am2ico*.tsv "${result_dir}/"

#XLWiC
## wic_english
lang="en"
types=("train" "valid")
files=("train" "dev")
for ((i = 0; i < ${#types[@]}; i++));
do
	type="${types[i]}"
	file="${files[i]}"
	lang_dir="${xlwic_dir}/wic_english"
	python3 preprocess_wic.py \
		--xlwic_path "${lang_dir}/${type}_${lang}.txt" \
		--filename "xlwic.${file}.${lang}-${lang}"
done

## xlwic_wikt
names=("french" "german" "italian")
langs=("fr" "de" "it")
types=("train" "valid" "test")
files=("train" "dev" "test")
for ((i = 0; i < ${#langs[@]}; i++));
do
	name="${names[i]}"
	lang="${langs[i]}"
	lang_dir="${xlwic_dir}/xlwic_wikt/${name}_${lang}"

	paste "${lang_dir}/${lang}_test_data.txt" "${lang_dir}/${lang}_test_gold.txt" > "${lang_dir}/${lang}_test.txt"

	for ((j = 0; j < ${#types[@]}; j++));
	do
		type="${types[j]}"
		file="${files[j]}"
		python3 preprocess_wic.py \
			--xlwic_path "${lang_dir}/${lang}_${type}.txt" \
			--filename "xlwic.${file}.${lang}-${lang}"
	done
	paste "${lang_dir}/IV/${lang}_iv_test_data.txt" "${lang_dir}/IV/${lang}_iv_test_gold.txt" > "${lang_dir}/${lang}_iv_test.txt"
	paste "${lang_dir}/OOV/${lang}_oov_test_data.txt" "${lang_dir}/OOV/${lang}_oov_test_gold.txt" > "${lang_dir}/${lang}_oov_test.txt"
	python3 preprocess_wic.py \
		--xlwic_path "${lang_dir}/${lang}_iv_test.txt" \
		--filename "xlwic.iv-test.${lang}-${lang}"
	python3 preprocess_wic.py \
		--xlwic_path "${lang_dir}/${lang}_oov_test.txt" \
		--filename "xlwic.oov-test.${lang}-${lang}"
done

## xlwic_wn
names=("bulgarian" "croatian" "dutch" "farsi" "korean" "chinese" "danish" "estonian" "japanese")
langs=("bg" "hr" "nl" "fa" "ko" "zh" "da" "et" "ja")
types=("valid" "test")
files=("dev" "test")
for ((i = 0; i < ${#langs[@]}; i++));
do
	name="${names[i]}"
	lang="${langs[i]}"
	lang_dir="${xlwic_dir}/xlwic_wn/${name}_${lang}"

	paste "${lang_dir}/${lang}_test_data.txt" "${lang_dir}/${lang}_test_gold.txt" > "${lang_dir}/${lang}_test.txt"

	for ((j = 0; j < ${#types[@]}; j++));
	do
		type="${types[j]}"
		file="${files[j]}"
		python3 preprocess_wic.py \
			--xlwic_path "${lang_dir}/${lang}_${type}.txt" \
			--filename "xlwic.${file}.${lang}-${lang}"
	done
done

mv preprocess_wic_debug.log "${result_dir}/"
mv xlwic*.tsv "${result_dir}/"

#MCL-WiC
## training
lang_dir="${mclwic_dir}/training"
python3 preprocess_wic.py \
	--mclwic_data "${lang_dir}/training.en-en.data" \
	--mclwic_gold "${lang_dir}/training.en-en.gold" \
	--filename "mclwic.train.en-en"

## dev
lang_dir="${mclwic_dir}/dev/multilingual"
langs="ar en fr ru zh"
for lang in ${langs};
do
	python3 preprocess_wic.py \
		--mclwic_data "${lang_dir}/dev.${lang}-${lang}.data" \
		--mclwic_gold "${lang_dir}/dev.${lang}-${lang}.gold" \
		--filename "mclwic.dev.${lang}-${lang}"
done

## test
lang_dir="${mclwic_dir}/test/multilingual"
langs="ar en fr ru zh"
for lang in ${langs};
do
	echo test.${lang}-${lang}
	python3 preprocess_wic.py \
		--mclwic_data "${lang_dir}/test.${lang}-${lang}.data" \
		--mclwic_gold "${lang_dir}/test.${lang}-${lang}.gold" \
		--filename "mclwic.test.${lang}-${lang}"
done

lang_dir="${mclwic_dir}/test/crosslingual"
langs="ar fr ru zh"
for lang in ${langs};
do
	echo test.en-${lang}
	python3 preprocess_wic.py \
		--mclwic_data "${lang_dir}/test.en-${lang}.data" \
		--mclwic_gold "${lang_dir}/test.en-${lang}.gold" \
		--filename "mclwic.test.en-${lang}"
done

mv preprocess_wic_debug.log "${result_dir}/"
mv mclwic*.tsv "${result_dir}/"
