#!/bin/bash
python -u src/train.py --data_dir en-de --spm_en en-de/sp_en.model --spm_de en-de/sp_de.model --epochs 4 --batch_size 64
