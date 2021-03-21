#!/bin/bash

cd `dirname $0`

SHELL=/bin/bash
# パスを通す
source /home/ec2-user/.bashrc
# 好きなPython環境を設定
source activate tensorflow2_latest_serving

cd /home/ec2-user/numerai
python3 predict_signals.py

sudo shutdown +5