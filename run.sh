#! /bin/sh
#
# run.sh
# Copyright (C) 2020 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#


CUDA_VISIBLE_DEVICES=`free-gpu` python compute_internal_pd.py examples/translation/wmt17_en_de/int_sa_kasai_official_make/ --path examples/translation/checkpoints/trans_ende_12-1_0.2/checkpoint_top5_average.pt --source-lang en --target-lang de --out examples/translation/wmt17_en_de/int_sa_kasai_official_make/pd
cd examples/translation/wmt17_en_de/int_sa_kasai_official_make/plot
python ../../print_sa.py ../pd.pd > pd
