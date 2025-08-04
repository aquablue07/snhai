{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/bash\
# This script runs the entire workflow for the loan adjudication project.\
# It will exit immediately if any command fails.\
set -e\
\
echo "--- Step 1 of 3: Generating Synthetic Data ---"\
python synthetic_data.py\
\
echo "\\n--- Step 2 of 3: Fine-Tuning FinBERT Model ---"\
python finbert_finetune_lora.py\
\
echo "\\n--- Step 3 of 3: Evaluating Model and Generating Report ---"\
python eval_by_importances.py\
\
echo "\\n--- Workflow Complete ---"}