# 📩 SMS Spam Detector

This is a Streamlit web app that detects spam messages using a trained machine learning model.

## Contents
- `app.py` - Streamlit application (single message prediction & dataset evaluation)
- `requirements.txt` - Dependencies
- `README.md` - This file
- `PLACE_MODEL_HERE.txt` - Instructions about placing your trained model
- `sample_test.csv` - Small sample CSV you can use to test the Evaluation mode

## How to use
1. Train your model in Colab and export it:
   ```python
   import joblib
   joblib.dump(model, "sms_spam_model.pkl")
   ```
2. Download `sms_spam_model.pkl` and place it in this project's root folder (next to `app.py`).
3. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud
1. Push the repository (including `sms_spam_model.pkl`) to GitHub.
2. On https://share.streamlit.io, create a new app and point it to your GitHub repo and `app.py`.