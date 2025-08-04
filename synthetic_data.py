import json
import random
import pandas as pd

def load_rules(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['personal_loan_credit_rules']['rules']

def create_applicant():
    return {
        "age": random.randint(16, 70),
        "credit_score": random.randint(500, 850),
        "annual_income_usd": random.randint(15000, 150000),
        "debt_to_income_ratio_percent": random.randint(10, 70),
        "employment_status": random.choice(["employed_full_time", "employed_part_time", "unemployed", "self_employed", "retired"]),
        "current_employment_duration_months": random.randint(0, 120),
        "residency_status": random.choice(["US_Citizen", "Permanent_Resident", "Visa_Holder"]),
        "has_bankruptcy_recent": random.choice([True, False]),
        "has_verifiable_bank_account": random.choice([True, False])
    }

def create_hard_case():
    applicant = {
        "age": random.randint(25, 60),
        "credit_score": random.randint(700, 850),
        "annual_income_usd": random.randint(40000, 60000),
        "debt_to_income_ratio_percent": random.randint(20, 35),
        "employment_status": "employed_full_time",
        "current_employment_duration_months": random.randint(24, 120),
        "residency_status": "US_Citizen",
        "has_bankruptcy_recent": True,
        "has_verifiable_bank_account": True
    }
    loan = {"requested_amount_usd": applicant["annual_income_usd"] * 0.7}
    return applicant, loan

def evaluate(application, rules):
    flag_reasons = []
    reject_reasons = [] # Will store tuples of (severity, name)

    for rule in rules:
        field_path = rule['field'].split('.')
        value = application
        for part in field_path: value = value.get(part, {})
        if value == {}: return "REJECT", "Missing field"
        if 'value_field_multiplier' in rule:
            base = application
            for part in rule['value_field_multiplier'].split('.'): base = base.get(part, 0)
            rule_value = base * rule['multiplier_value']
        else:
            rule_value = rule['value']
        fail = False
        op = rule['operator']
        if op == '>=' and not value >= rule_value: fail = True
        elif op == '<=' and not value <= rule_value: fail = True
        elif op == 'in' and value not in rule_value: fail = True
        elif op == 'is' and value is not rule_value: fail = True

        if fail:
            action = rule['action_on_fail']
            if action == "REJECT":
                reject_reasons.append((rule['severity'], rule['name']))
            elif action == "FLAG_REVIEW":
                flag_reasons.append(rule['name'])

    if reject_reasons:
        critical_rejects = [r for sev, r in reject_reasons if sev == 'CRITICAL']
        if critical_rejects:
            return "REJECT", critical_rejects[0]
        return "REJECT", reject_reasons[0][1]

    if flag_reasons:
        return "FLAG_REVIEW", "; ".join(flag_reasons)

    return "APPROVE", "Key requirements met"

def format_input(app, loan):
    return (
        f"age: {app['age']}, credit_score: {app['credit_score']}, income: {app['annual_income_usd']}, "
        f"dti: {app['debt_to_income_ratio_percent']}, employment: {app['employment_status']}, "
        f"employment_months: {app['current_employment_duration_months']}, residency: {app['residency_status']}, "
        f"bankruptcy: {app['has_bankruptcy_recent']}, bank_account: {app['has_verifiable_bank_account']}, "
        f"loan_amount: {loan['requested_amount_usd']}"
    )

def generate_data(rules, n=2000, target_ratio={"APPROVE": 0.33, "REJECT": 0.33, "FLAG_REVIEW": 0.34}):
    data = []
    counts = {k: 0 for k in target_ratio}
    target_counts = {k: int(target_ratio[k] * n) for k in target_ratio}

    while sum(counts.values()) < n:
        if random.random() < 0.15 and counts["REJECT"] < target_counts["REJECT"]:
             applicant, loan = create_hard_case()
        else:
            applicant = create_applicant()
            loan = {"requested_amount_usd": random.randint(5000, 100000)}
        
        application = {"applicant": applicant, "loan_application": loan}
        decision, reason = evaluate(application, rules)

        if decision not in counts:
            continue

        if counts[decision] < target_counts[decision]:
            input_text = "classify: " + format_input(applicant, loan)
            output_text = f"{decision} – {reason}"
            data.append({"input_text": input_text, "output_text": output_text})
            counts[decision] += 1

    print(" Class distribution:", counts)
    return data

def main():
    rules = load_rules("fine_tune_llm_credit_rules.json")
    data = generate_data(rules, n=2000)

    df = pd.DataFrame(data)
    train = df.sample(frac=0.8, random_state=42)
    val = df.drop(train.index)

    train.to_json("train_data.jsonl", orient="records", lines=True)
    val.to_json("val_data.jsonl", orient="records", lines=True)
    print(" Dataset generated and saved as train_data.jsonl / val_data.jsonl")

if __name__ == "__main__":
    main()