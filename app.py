import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Dataset
@st.cache_data
def load_data():
    file_path = "StudentPerformanceFactors.csv"  # Replace with your dataset path
    return pd.read_csv(file_path)

# Load data
data = load_data()

# Preprocessing Data
relevant_columns = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity", "Exam_Score"
]
filtered_data = data[relevant_columns].dropna()

# Features and Target for Regression
X_regression = filtered_data.drop(columns=["Exam_Score"])
y_regression = filtered_data["Exam_Score"]

# Features and Target for Classification (Pass/Fail)
y_classification = (filtered_data["Exam_Score"] >= 70).astype(int)
X_classification = filtered_data[["Attendance", "Previous_Scores", "Exam_Score"]]

# Train-Test Split for Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Train-Test Split for Classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

# Scaling for Regression
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Models
# Regression Model
model_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
model_regressor.fit(X_train_reg_scaled, y_train_reg)

# Classification Model
model_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
model_classifier.fit(X_train_clf, y_train_clf)

# Tambahkan kategori ke dataset sebelum bagian Feature Importance
study_bins = [0, 10, 20, 30, 40]
study_labels = ['Low', 'Moderate', 'Good', 'Excellent']
activity_bins = [0, 2, 4, 6, 8]
activity_labels = ['Low', 'Moderate', 'Good', 'Excellent']

filtered_data['Study_Habits_Level'] = pd.cut(
    filtered_data['Hours_Studied'], bins=study_bins, labels=study_labels, include_lowest=True
)
filtered_data['Physical_Activity_Level'] = pd.cut(
    filtered_data['Physical_Activity'], bins=activity_bins, labels=activity_labels, include_lowest=True
)

# Streamlit Dashboard
menu = st.sidebar.selectbox("Select Analysis:", [
    "Data Overview", 
    "Data Visualization", 
    "Exam Score Prediction", 
    "Pass/Fail Prediction", 
    "Feature Importance", 
    "Model Evaluation"
])

# Data Overview
if menu == "Data Overview":
    st.title("🎓 Student Performance Insights Dashboard")  # Only show this title here
    st.header("📋 Data Overview")
    st.dataframe(data)
    st.write("**Dataset Summary:**")
    st.write(data.describe())

# Data Visualization
elif menu == "Data Visualization":
    st.header("📊 Data Visualization")

    # Interactive Histogram of Exam Score Distribution
    fig_histogram = px.histogram(filtered_data, x='Exam_Score', nbins=30, title="Distribution of Exam Scores",
                                  labels={'Exam_Score': 'Exam Score'}, opacity=0.75)
    fig_histogram.update_layout(
        xaxis_title="Exam Score",
        yaxis_title="Frequency",
        bargap=0.1  # Adjust the gap between bars
    )
    st.plotly_chart(fig_histogram)

    # Interactive Bar Plot for Average Exam Score by Tutoring Sessions
    avg_score = filtered_data.groupby("Tutoring_Sessions")["Exam_Score"].mean().reset_index()
    fig_avg_score = px.bar(avg_score, x="Tutoring_Sessions", y="Exam_Score", title="Average Exam Score by Tutoring Sessions",
                           labels={'Tutoring_Sessions': 'Tutoring Sessions', 'Exam_Score': 'Average Exam Score'},
                           color="Exam_Score", color_continuous_scale='Viridis')
    fig_avg_score.update_layout(
        xaxis_title="Tutoring Sessions",
        yaxis_title="Average Exam Score"
    )
    st.plotly_chart(fig_avg_score)

    # Interactive Line Plot for Hours Studied vs Exam Score
    # Create a sample dataset to ensure clarity
    sampled_data = filtered_data.sample(n=50, random_state=42)  # Adjust sample size as needed
    sampled_data = sampled_data.sort_values(by="Hours_Studied")  # Sort data for better visualization

    # Line plot using Plotly
    fig_line = px.line(
        sampled_data, 
        x="Hours_Studied", 
        y="Exam_Score", 
        title="Hours Studied vs Exam Score (Sampled Data)",
        labels={"Hours_Studied": "Hours Studied", "Exam_Score": "Exam Score"},
        markers=True  # Add markers for data points
    )
    fig_line.update_traces(line=dict(width=2))  # Customize line width
    fig_line.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),  # Set tick intervals
        yaxis=dict(title="Exam Score"),
        hovermode="x unified"  # Show values in hover tooltip
    )
    st.plotly_chart(fig_line)

    # New Plot: Bar Plot for Exam Scores
    fig, ax = plt.subplots(figsize=(15, 5))  # Adjust figure size for Streamlit
    exam_score_counts = filtered_data['Exam_Score'].value_counts(dropna=False)
    exam_score_counts.plot.bar(ax=ax, color="skyblue")
    ax.set_title("Comparison of Exam Scores")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Exam Score Prediction
elif menu == "Exam Score Prediction":
    st.header("📈 Exam Score Prediction")
    hours_studied = st.slider("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=80)
    sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    previous_scores = st.slider("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
    tutoring_sessions = st.slider("Tutoring Sessions", min_value=0, max_value=10, value=2)
    physical_activity = st.slider("Physical Activity (hours per week)", min_value=0, max_value=20, value=5)

    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Tutoring_Sessions": [tutoring_sessions],
        "Physical_Activity": [physical_activity],
    })
    if st.button("Predict Exam Score"):
        input_scaled = scaler.transform(input_data)
        prediction = model_regressor.predict(input_scaled)
        st.subheader(f"Predicted Exam Score: {prediction[0]:.2f}")

    # Exam Score Distribution
    # Sort the dataset by Exam Scores and pick the top 5 and bottom 5
    top_5_scores = filtered_data.nlargest(5, 'Exam_Score')[['Exam_Score']]
    bottom_5_scores = filtered_data.nsmallest(5, 'Exam_Score')[['Exam_Score']]

    # Plot Top 5 Highest Exam Scores
    fig_top = px.bar(top_5_scores, x=top_5_scores.index, y='Exam_Score', title="Top 5 Highest Exam Scores",
                     labels={'Exam_Score': 'Exam Score', 'index': 'No'},
                     color='Exam_Score', color_continuous_scale='Viridis', orientation='v')
    st.plotly_chart(fig_top)

    # Plot Top 5 Lowest Exam Scores
    fig_bottom = px.bar(bottom_5_scores, x=bottom_5_scores.index, y='Exam_Score', title="Top 5 Lowest Exam Scores",
                        labels={'Exam_Score': 'Exam Score', 'index': 'No'},
                        color='Exam_Score', color_continuous_scale='Reds', orientation='v')
    st.plotly_chart(fig_bottom)

# Pass/Fail Prediction
elif menu == "Pass/Fail Prediction":
    st.header("🎯 Pass/Fail Prediction")
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=80)
    previous_scores = st.slider("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
    exam_score = st.slider("Exam Score", min_value=0.0, max_value=100.0, value=65.0)

    input_data_clf = pd.DataFrame({
        "Attendance": [attendance],
        "Previous_Scores": [previous_scores],
        "Exam_Score": [exam_score],
    })

    if st.button("Predict Pass/Fail"):
        prediction_clf = model_classifier.predict(input_data_clf)
        status = "Pass" if prediction_clf[0] == 1 else "Fail"
        st.subheader(f"Prediction: {status}")

    # Logistic Regression for Pass/Fail Prediction
    # Add the "Pass" column to the DataFrame
    filtered_data['Pass'] = (filtered_data['Exam_Score'] >= 70).astype(int)

    # Features and labels for Logistic Regression
    X_logistic = filtered_data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']]
    y_logistic = filtered_data['Pass']

    # Train-test split for Logistic Regression
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

    # Initialize Logistic Regression model
    logistic_model = LogisticRegression()

    # Train the model
    logistic_model.fit(X_train_log, y_train_log)

    # Predict using the logistic model
    y_pred_log = logistic_model.predict(X_test_log)

    # Interactive Pie Chart: Distribution of Pass/Fail Predictions
    # Count the occurrences of Pass (1) and Fail (0)
    pass_fail_counts = pd.Series(y_pred_log).value_counts()
    pass_fail_labels = ['Fail', 'Pass']  # Ensure "Fail" is first
    pass_fail_data = pd.DataFrame({
        'Prediction': pass_fail_labels,
        'Count': pass_fail_counts
    })

    # Customize color for Pass and Fail (green pastel for Pass, red pastel for Fail)
    color_map = {"Pass": "#8FBC8F", "Fail": "#FF6F61"}  # Green pastel and Red pastel

    # Plot interactive pie chart with custom colors
    fig = px.pie(pass_fail_data, names='Prediction', values='Count', title="Distribution of Pass/Fail Predictions", color='Prediction', 
                 color_discrete_map=color_map)
    st.plotly_chart(fig)

# Feature Importance
elif menu == "Feature Importance":
    st.header("🔍 Feature Importance")

    # Get feature importances from the Random Forest Regressor model
    feature_importances = model_regressor.feature_importances_
    features = X_regression.columns

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)  # Sort ascending for horizontal bars

    # Create an interactive horizontal bar chart with Plotly
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                 title="Feature Importance that Affect Exam Scores",
                 labels={'Feature': 'Feature', 'Importance': 'Importance Score'},
                 color='Importance', color_continuous_scale='Viridis',
                 text='Importance', orientation='h')

    # Update layout for better visibility
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside',
                      marker=dict(line=dict(color='black', width=1)))

    # Show the plot in Streamlit
    st.plotly_chart(fig)

    # Tambahkan Heatmap interaktif dari analisis Google Colab
    
    kelompok_data = filtered_data.groupby(['Study_Habits_Level', 'Physical_Activity_Level'])['Exam_Score'].mean().reset_index()
    heatmap_data = kelompok_data.pivot(index='Study_Habits_Level', columns='Physical_Activity_Level', values='Exam_Score')

    # Visualisasi heatmap interaktif menggunakan Plotly
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Average Exam Score by Study Habits and Physical Activity Levels",
        labels={'x': "Physical Activity Level", 'y': "Study Habits Level", 'color': "Average Exam Score"},
        color_continuous_scale="YlGnBu"
    )
    fig_heatmap.update_layout(
        xaxis_title="Physical Activity Level",
        yaxis_title="Study Habits Level",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig_heatmap)

    # Tambahkan Penjelasan Kategori
    st.write("**Category Explanations:**")
    st.write("**Hours Studied:**")
    st.write("0-10: Low, 11-20: Moderate, 21-30: Good, 31-40: Excellent")
    st.write("**Physical Activity (Hours):**")
    st.write("0-2: Low, 3-4: Moderate, 5-6: Good, 7-8: Excellent")

# Model Evaluation
elif menu == "Model Evaluation":
    st.header("📊 Model Evaluation")

    # Regression Evaluation
    y_pred_reg = model_regressor.predict(X_test_reg_scaled)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    st.subheader("Regression Model Evaluation")
    st.write(f"R2 Score: {r2:.2f}")
    st.write(f"Mean Absolute Error: {mae:.2f}")

    # Classification Evaluation
    y_pred_clf = model_classifier.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    st.subheader("Classification Model Evaluation")
    st.write(f"Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test_clf, y_pred_clf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)