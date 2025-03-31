import pandas as pd
import numpy as np

# Load the datasets
matches_df = pd.read_csv('matches.csv')
deliveries_df = pd.read_csv('deliveries.csv')

# Handle missing values (e.g., fill or drop rows/columns with missing values)
matches_df.fillna({'toss_winner': 'Unknown', 'winner': 'Unknown'}, inplace=True)

# Feature Engineering: Calculate team performance metrics
matches_df['team1_win_percentage'] = matches_df.groupby('team1')['winner'].apply(lambda x: (x == 'team1').mean())
matches_df['team2_win_percentage'] = matches_df.groupby('team2')['winner'].apply(lambda x: (x == 'team2').mean())

# Toss win percentage per team
matches_df['toss_win_percentage'] = matches_df.groupby('toss_winner')['toss_winner'].transform('count') / len(matches_df) 
import matplotlib.pyplot as plt
import seaborn as sns

# Toss wins per team
toss_wins = matches_df['toss_winner'].value_counts()
sns.barplot(x=toss_wins.index, y=toss_wins.values)
plt.title('Toss Wins by Team')
plt.xticks(rotation=90)
plt.show()

# Match wins per team
match_wins = matches_df['winner'].value_counts()
sns.barplot(x=match_wins.index, y=match_wins.values)
plt.title('Match Wins by Team')
plt.xticks(rotation=90)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare the features and target for Toss Winner Prediction
features = ['team1_win_percentage', 'team2_win_percentage', 'toss_win_percentage', 'home_team']
target = 'toss_winner'

# Train-Test Split
X = matches_df[features]
y = matches_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Classifier
toss_model = RandomForestClassifier(n_estimators=100, random_state=42)
toss_model.fit(X_train, y_train)

# Model Evaluation
y_pred = toss_model.predict(X_test)
print(f"Toss Winner Prediction Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred)) # Prepare the features and target for Match Winner Prediction
features = ['team1_win_percentage', 'team2_win_percentage', 'toss_win_percentage', 'home_team']
target = 'winner'

# Train-Test Split
X = matches_df[features]
y = matches_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Classifier
match_model = RandomForestClassifier(n_estimators=100, random_state=42)
match_model.fit(X_train, y_train)

# Model Evaluation
y_pred = match_model.predict(X_test)
print(f"Match Winner Prediction Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred)) # Example of predicting for IPL 2025
ipl_2025_schedule = pd.read_csv('ipl_2025_schedule.csv')  # Make sure to have this file with match details

# Assume the same features for predictions
predictions = []

for index, match in ipl_2025_schedule.iterrows():
    match_features = {
        'team1_win_percentage': matches_df[matches_df['team1'] == match['team1']]['team1_win_percentage'].mean(),
        'team2_win_percentage': matches_df[matches_df['team2'] == match['team2']]['team2_win_percentage'].mean(),
        'toss_win_percentage': matches_df[matches_df['toss_winner'] == match['team1']]['toss_win_percentage'].mean(),
        'home_team': 1 if match['team1'] == match['city'] else 0
    }
    match_df = pd.DataFrame([match_features])
    
    toss_prediction = toss_model.predict(match_df)
    match_prediction = match_model.predict(match_df)

    predictions.append({
        'Match': f"{match['team1']} vs {match['team2']}",
        'Toss Winner': toss_prediction[0],
        'Match Winner': match_prediction[0]
    })

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)
print(predictions_df)
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, toss_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, color='b', label="Toss Winner ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", color="r")
plt.title("ROC-AUC Curve for Toss Winner Prediction")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

roc_auc = roc_auc_score(y_test, toss_model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_auc}")