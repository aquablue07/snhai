# Hybrid LLM for Loan Adjudication

This project demonstrates a hybrid AI system for adjudicating personal loan applications based on a deterministic set of business rules. It uses a fine-tuned financial language model (FinBERT) to make classification decisions and a rule-based system to provide auditable explanations.

## Features

 Hybrid AI Model : Uses a fine-tuned FinBERT model for classification, providing flexibility for future enhancements.
Rule-Based Explainability: Generates explanations that are 100% compliant with the provided business logic for auditability.
Explainable AI (XAI) : The evaluation script includes a feature importance analysis to show which input features most influenced the model's decision.
Synthetic Data Generation**: Includes a robust script to generate a balanced, high-quality dataset for training.

## Requirements

* Python 3.10+
* Conda (or another virtual environment manager)

## Setup

1.  Create and Activate a New Conda Environment:**
    ```bash
    conda create --name loan_adjudication_env python=3.10 -y
    conda activate loan_adjudication_env
    ```

2. Install Required Libraries:
    This command installs a known-compatible set of all necessary libraries.
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers==4.42.4 accelerate==0.31.0 peft==0.11.1 datasets scikit-learn pandas matplotlib seaborn
    ```

## Usage

### Automated Workflow
The entire workflow (data generation, training, and evaluation) can be run with a single command.

1.  **Make the script executable (only needs to be done once):**
    ```bash
    chmod +x run_workflow.sh
    ```
2.  **Run the workflow:**
    ```bash
    ./run_workflow.sh
    ```

### Running Steps Manually
If you prefer to run each step individually:

1.  **Generate the Dataset:**
    ```bash
    python synthetic_data.py
    ```
2.  **Train the Model:**
    ```bash
    python finbert_finetune_lora.py
    ```
3.  **Evaluate the Model:**
    ```bash
    python eval_by_importances.py
    ```

## File Descriptions

* `run_workflow.sh`: An executable script to run the entire project pipeline.
* `synthetic_data.py`: Script to generate the training and validation datasets.
* `finbert_finetune_lora.py`: Script to fine-tune the FinBERT model using LoRA.
* `eval_by_importances.py`: Script to evaluate the trained model and analyze its reasoning.}
