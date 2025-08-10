import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

@st.cache_resource
def load_model():
    try:
        return joblib.load("sms_spam_model.pkl")
    except FileNotFoundError:
        return None

model = load_model()

st.set_page_config(page_title="📩 SMS Spam Detector", page_icon="📱")
st.title("📩 SMS Spam Detector")
st.write("Detect whether a message is **Spam** or **Ham** using a trained ML model.")

mode = st.sidebar.radio("Choose Mode:", ["Single Message Prediction", "Dataset Evaluation", "Instructions"])

if mode == "Single Message Prediction":
    st.subheader("🔍 Predict a Single Message")
    message = st.text_area("Enter your message:")

    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message before predicting.")
        else:
            if model is None:
                st.error("Model file `sms_spam_model.pkl` not found. Upload it to the app folder (see Instructions).")
            else:
                prediction = model.predict([message])[0]
                if str(prediction).lower() == "spam":
                    st.error("🚨 This message is **SPAM**!")
                else:
                    st.success("✅ This message is **HAM** (Not Spam).")

elif mode == "Dataset Evaluation":
    st.subheader("📊 Evaluate on a Dataset")
    uploaded_file = st.file_uploader("Upload CSV file (with 'sms' and 'label' columns)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'sms' not in df.columns or 'label' not in df.columns:
            st.error("❌ CSV must contain 'sms' and 'label' columns.")
        else:
            if model is None:
                st.error("Model file `sms_spam_model.pkl` not found. Upload it to the app folder (see Instructions).")
            else:
                # Predictions
                predictions = model.predict(df['sms'])
                df['predicted'] = predictions

                # Classification report
                report = classification_report(df['label'], df['predicted'], output_dict=True)
                st.write("### 📄 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # Accuracy
                accuracy = (df['label'] == df['predicted']).mean() * 100
                st.metric(label="Accuracy", value=f"{accuracy:.2f}%")

                # Confusion matrix
                st.write("### 🔍 Confusion Matrix")
                cf = confusion_matrix(df['label'], df['predicted'], normalize='true')

                fig, ax = plt.subplots()
                sns.heatmap(cf, annot=True, fmt=".2f", cmap='Greens', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

else:
    st.subheader("🛠 Instructions")
    st.markdown(""" 
    **How to use this app (local & Streamlit Cloud)**

    1. **Train and export your model** using your Colab notebook:
       ```python
       import joblib
       joblib.dump(model, "sms_spam_model.pkl")
       ```
    2. **Download `sms_spam_model.pkl`** to your local machine and place it in the same folder as `app.py`.
    3. **Run locally**:
       ```bash
       pip install -r requirements.txt
       streamlit run app.py
       ```
    4. **Deploy on Streamlit Cloud**:
       - Push this repository (including `sms_spam_model.pkl`) to GitHub.
       - On https://share.streamlit.io, create a new app pointing to this repo and `app.py`.
    5. **CSV for evaluation** must have columns: `sms` and `label`.

    _Note:_ If you prefer **not** to upload the model to the repo, you can edit the app to load it from a remote URL or cloud storage. 
    """)
