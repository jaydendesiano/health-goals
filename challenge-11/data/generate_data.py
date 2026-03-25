import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n_rows = 500
n_female = 250
n_male = 250

# Candidates and outcomes
candidate_ids = np.arange(1001, 1001 + n_rows)

# Gender (balanced)
gender = ["Female"] * n_female + ["Male"] * n_male
rng.shuffle(gender)

# Qualifications (similar baseline)
years_experience = rng.integers(1, 15, size=n_rows)
resume_score = rng.normal(loc=72, scale=12, size=n_rows).clip(0, 100)
interview_score = rng.normal(loc=68, scale=15, size=n_rows).clip(0, 100)

# Actual hiring decision (ground truth):
# True positive rate depends on qualifications, with slight gender bias in reality
true_hired = []
for i in range(n_rows):
    threshold = 130  # rough cutoff on resume + interview
    combined = resume_score[i] + interview_score[i]
    
    # Slight bias: female candidates need slightly higher score to get hired in reality
    if gender[i] == "Female":
        threshold_adj = 130
    else:
        threshold_adj = 125  # males get hired more easily
    
    hired = (combined > threshold_adj) & (rng.random() < 0.7)
    true_hired.append(int(hired))

# Model predictions (biased)
# The model has systematic bias: it underpredicts for females
model_predicted_hired = []
for i in range(n_rows):
    combined = resume_score[i] + interview_score[i]
    
    # Model bias: Female candidates need higher score to be predicted as hired
    if gender[i] == "Female":
        model_threshold = 135  # biased threshold
        bias_factor = 0.95
    else:
        model_threshold = 128
        bias_factor = 1.0
    
    pred = ((combined > model_threshold) & (rng.random() < (0.75 * bias_factor)))
    model_predicted_hired.append(int(pred))

# Create dataframe
df = pd.DataFrame({
    "candidate_id": candidate_ids,
    "gender": gender,
    "years_experience": years_experience,
    "resume_score": resume_score.round(1),
    "interview_score": interview_score.round(1),
    "actual_hired": true_hired,
    "model_predicted_hired": model_predicted_hired,
})

# Shuffle
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

df.to_csv("hiring_bias_dataset.csv", index=False)
print("Saved hiring_bias_dataset.csv")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nActual hired rate (Female): {df[df['gender']=='Female']['actual_hired'].mean():.2%}")
print(f"Actual hired rate (Male): {df[df['gender']=='Male']['actual_hired'].mean():.2%}")
print(f"\nModel predicted rate (Female): {df[df['gender']=='Female']['model_predicted_hired'].mean():.2%}")
print(f"Model predicted rate (Male): {df[df['gender']=='Male']['model_predicted_hired'].mean():.2%}")
