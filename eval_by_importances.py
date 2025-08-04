import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# config
MODEL_PATH = "finbert-loan-classifier"
LABELS = ["APPROVE", "REJECT", "FLAG_REVIEW"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() 

val_dataset = load_dataset("json", data_files={"val": "val_data.jsonl"})["val"]

with open("fine_tune_llm_credit_rules.json", 'r') as f:
    rules = json.load(f)['personal_loan_credit_rules']['rules']

# get metrics
print("Running full evaluation for metrics...")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

y_true = []
y_pred = []

for row in tqdm(val_dataset, desc="Evaluating"):
    true_label_str = row["output_text"].split(" – ")[0]
    y_true.append(true_label_str)
    
    prediction = classifier(row["input_text"])[0]
    predicted_label_str = prediction['label']
    y_pred.append(predicted_label_str)

#  Display Metrics
print("\n Classification Report:")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

print("\n Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#  Manual Feature Importance Analysis -- alternative to SHAP
print("\n Generating Sample Predictions with Manual Feature Importance...")

def get_word_importance(text, predicted_label, classifier, top_k=5):
    """
    Get word importance by removing each word and measuring prediction change
    
    Args:
        text: Input text to analyze
        predicted_label: The predicted label for the original text
        classifier: The classification pipeline
        top_k: Number of top important words to return
    
    Returns:
        List of (word, importance_score) tuples sorted by importance
    """
    words = text.split()
    
    # Get baseline prediction confidence
    baseline_pred = classifier(text)[0]
    baseline_score = baseline_pred['score'] if baseline_pred['label'] == predicted_label else 0
    
    word_importance = []
    
    for i, word in enumerate(words):
        # Create text without this word
        masked_text = " ".join(words[:i] + words[i+1:])
        
        if masked_text.strip():  # Only if there's still text left
            try:
                masked_pred = classifier(masked_text)[0]
                # Calculate importance as change in confidence for the predicted label
                if masked_pred['label'] == predicted_label:
                    masked_score = masked_pred['score']
                else:
                    masked_score = 0  # Label changed, so original word was very important
                
                importance = baseline_score - masked_score
                word_importance.append((word, importance))
                
            except Exception as e:
                # If there's an error with the masked text, assign zero importance
                word_importance.append((word, 0.0))
        else:
            # If removing this word results in empty text, it's very important
            word_importance.append((word, baseline_score))
    
    # Sort by absolute importance and return top_k
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_importance[:top_k]

def map_feature_to_reason(feature_name, rules, word_importance_list):
    """
    Map important words to business rules and generate explanation
    
    Args:
        feature_name: Most important word/feature
        rules: Business rules from JSON
        word_importance_list: List of (word, importance) tuples
    
    Returns:
        String explanation of the decision
    """
    # Clean the feature name
    clean_feature = feature_name.replace(":", "").replace(",", "").strip().lower()
    
    # Look for exact matches in business rules
    for rule in rules:
        rule_field = rule["field"].lower()
        if clean_feature in rule_field or any(part in clean_feature for part in rule_field.split("_")):
            return rule["name"]
    
    # If no direct rule match, try to infer from the word and context
    if clean_feature.isdigit():
        # Check if it's a numeric value that might relate to thresholds
        value = int(clean_feature)
        if value < 600:
            return "Credit Score Below Minimum"
        elif value > 100000:
            return "High Loan Amount"
        elif value > 50:
            return "High Debt-to-Income Ratio"
    
    # Check for specific keywords
    keyword_mappings = {
        'credit_score': 'Credit Score Requirements',
        'income': 'Income Verification',
        'dti': 'Debt-to-Income Ratio',
        'employment': 'Employment Status',
        'bankruptcy': 'Bankruptcy History',
        'bank_account': 'Bank Account Verification',
        'loan_amount': 'Loan Amount Limits',
        'age': 'Age Requirements',
        'residency': 'Residency Status',
        'true': 'Negative Flag Detected',
        'false': 'Missing Required Information'
    }
    
    for keyword, reason in keyword_mappings.items():
        if keyword in clean_feature:
            return reason
    
    # If still no match, use the most important words to create a general reason
    important_words = [word for word, _ in word_importance_list[:3]]
    if any(word.lower() in ['true', 'false'] for word in important_words):
        return "Risk Factor Assessment"
    
    return f"Model Decision Based on: {feature_name}"

def analyze_prediction_reasoning(text, classifier, rules):
    """
    Analyze a single prediction and provide reasoning
    
    Args:
        text: Input text to analyze
        classifier: Classification pipeline
        rules: Business rules
    
    Returns:
        Tuple of (predicted_label, confidence, reasoning, top_words)
    """
    # Get prediction
    prediction = classifier(text)[0]
    predicted_label = prediction['label']
    confidence = prediction['score']
    
    if predicted_label == "APPROVE":
        reasoning = "All key requirements met"
        top_words = []
    else:
        # Get word importance
        word_importance = get_word_importance(text, predicted_label, classifier)
        
        if word_importance:
            top_word = word_importance[0][0]
            reasoning = map_feature_to_reason(top_word, rules, word_importance)
            top_words = word_importance
        else:
            reasoning = "General model decision"
            top_words = []
    
    return predicted_label, confidence, reasoning, top_words

# Generate explanations for sample predictions
print("Analyzing samples with detailed reasoning...")

for i in range(100, 120):    # can change the idx from 0 till 400
    row = val_dataset[i]
    raw_input = row["input_text"]
    
    try:
        predicted_label, confidence, reasoning, top_words = analyze_prediction_reasoning(
            raw_input, classifier, rules
        )
        
        print(f"\ Input: {raw_input}")
        print(f" Predicted: {predicted_label} (confidence: {confidence:.3f}) – {reasoning}")
        print(f" Ground Truth: {row['output_text']}")
        
        if top_words:
            print(f" Top influential words: {', '.join([f'{word}({score:.3f})' for word, score in top_words[:3]])}")
            
    except Exception as e:
        print(f"\n▶ Input: {raw_input}")
        print(f" Analysis failed: {str(e)}")
        print(f" Ground Truth: {row['output_text']}")

# --- Detailed Analysis for One Example ---
print("\n🔍 Detailed Analysis for One Example:")

sample_idx = 105
sample_row = val_dataset[sample_idx]
sample_input = sample_row["input_text"]

try:
    predicted_label, confidence, reasoning, top_words = analyze_prediction_reasoning(
        sample_input, classifier, rules
    )
    
    print(f"Input: {sample_input}")
    print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
    print(f"Ground Truth: {sample_row['output_text']}")
    print(f"Reasoning: {reasoning}")
    
    if top_words:
        print("\nWord Importance Analysis:")
        print("=" * 50)
        for i, (word, importance) in enumerate(top_words, 1):
            print(f"{i:2d}. '{word:15s}' → {importance:+.4f}")
        
        # Show what happens when we remove the most important word
        most_important_word = top_words[0][0]
        words = sample_input.split()
        word_index = next((i for i, w in enumerate(words) if w == most_important_word), -1)
        
        if word_index != -1:
            modified_text = " ".join(words[:word_index] + words[word_index+1:])
            modified_pred = classifier(modified_text)[0]
            
            print(f"\n🔄 Removing most important word '{most_important_word}':")
            print(f"   Original: {predicted_label} ({confidence:.3f})")
            print(f"   Modified: {modified_pred['label']} ({modified_pred['score']:.3f})")
            print(f"   Impact: {confidence - modified_pred['score']:.3f} confidence change")
    
except Exception as e:
    print(f"Detailed analysis failed: {str(e)}")

# --- Summary Statistics --- uncomment for closer monitoring
# print("\n📈 Feature Importance Summary:")
# print("Analyzing feature importance across multiple samples...")

# feature_importance_summary = {}
# sample_size = min(50, len(val_dataset) - 100)  # Analyze 50 samples

# for i in tqdm(range(100, 100 + sample_size), desc="Analyzing samples"):
#     row = val_dataset[i]
#     try:
#         predicted_label, confidence, reasoning, top_words = analyze_prediction_reasoning(
#             row["input_text"], classifier, rules
#         )
        
#         # Collect top words for summary
#         for word, importance in top_words[:3]:  # Top 3 words per sample
#             clean_word = word.lower().replace(":", "").replace(",", "")
#             if clean_word not in feature_importance_summary:
#                 feature_importance_summary[clean_word] = []
#             feature_importance_summary[clean_word].append(abs(importance))
    
#     except Exception as e:
#         continue

# # Show most frequently important features
# if feature_importance_summary:
#     print("\nMost frequently important features:")
#     feature_stats = []
#     for feature, importance_list in feature_importance_summary.items():
#         if len(importance_list) >= 3:  # Only features that appear in multiple samples
#             avg_importance = np.mean(importance_list)
#             frequency = len(importance_list)
#             feature_stats.append((feature, avg_importance, frequency))
    
#     feature_stats.sort(key=lambda x: x[1] * x[2], reverse=True)  # Sort by avg_importance * frequency
    
#     for i, (feature, avg_importance, frequency) in enumerate(feature_stats[:10], 1):
#         print(f"{i:2d}. '{feature}' - Avg importance: {avg_importance:.3f}, Frequency: {frequency}")

print("\n✅ Analysis complete!")



# shap version - doesnt work on all environments, but faster #
# import json
# import numpy as np
# import torch
# import shap
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from datasets import load_dataset
# from tqdm import tqdm
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Config
# MODEL_PATH = "finbert-loan-classifier"
# LABELS = ["APPROVE", "REJECT", "FLAG_REVIEW"]
# label2id = {label: i for i, label in enumerate(LABELS)}
# id2label = {i: label for i, label in enumerate(LABELS)}

# # Load Model, Data, and Rules 
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval() 
# val_dataset = load_dataset("json", data_files={"val": "val_data.jsonl"})["val"]

# with open("fine_tune_llm_credit_rules.json", 'r') as f:
#     rules = json.load(f)['personal_loan_credit_rules']['rules']

# # get metrics
# print("Running full evaluation for metrics...")
# classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
# y_true = []
# y_pred = []

# for row in tqdm(val_dataset, desc="Evaluating"):
#     true_label_str = row["output_text"].split(" – ")[0]
#     y_true.append(true_label_str)
    
#     prediction = classifier(row["input_text"])[0]
#     predicted_label_str = prediction['label']
#     y_pred.append(predicted_label_str)

# # --- Display Metrics ---
# print("\n Classification Report:")
# print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

# print("\n Generating Confusion Matrix...")
# cm = confusion_matrix(y_true, y_pred, labels=LABELS)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # --- SHAP-Based Reasoning for Samples ---
# print("\n Generating Sample Predictions with SHAP-based Reasoning...")

# explainer = shap.Explainer(model, tokenizer)

# def map_feature_to_reason(feature_name, rules):
#     clean_feature = feature_name.replace(":", "").strip()
#     for rule in rules:
#         if clean_feature in rule["field"]:
#             return rule["name"]
#     return "General Reason"

# for i in range(100,120):
#     row = val_dataset[i]
#     raw_input = row["input_text"]
    
#     prediction = classifier(raw_input)[0]
#     predicted_label = prediction['label']
    
#     generated_reason = "N/A"
#     if predicted_label != "APPROVE":
#         try:
#             # Manually tokenize and move to the correct device
#             inputs = tokenizer(raw_input, return_tensors="pt", padding=True, truncation=True, max_length=256)
#             device = model.device
#             inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            
#             # Pass the device-matched, tokenized inputs to the explainer
#             shap_values = explainer(inputs_on_device)
            
#             predicted_class_id = label2id[predicted_label]
#             class_specific_shap_values = shap_values.values[0][:, predicted_class_id]

#             tokens = shap_values.data[0]
            
#             top_feature_index = np.argmax(class_specific_shap_values)
            
#             if top_feature_index > 0:
#                 top_feature_name = tokens[top_feature_index - 1]
#             else:
#                 top_feature_name = tokens[top_feature_index]
            
#             generated_reason = map_feature_to_reason(top_feature_name, rules)
#         except IndexError as e:
#             generated_reason = f"SHAP failed with an error: {e}"
#         except Exception as e:
#             generated_reason = f"An unexpected error occurred: {e}"
#     else:
#         generated_reason = "Key requirements met"

#     print(f"\ Input: {raw_input}")
#     print(f" Predicted: {predicted_label} – {generated_reason}")
#     print(f" Ground Truth: {row['output_text']}")

