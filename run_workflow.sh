 #!/bin/bash\
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
