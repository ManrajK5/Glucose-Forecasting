# Glucose Forecasting

Machine learning project for forecasting blood glucose 30, 60, and 120 minutes ahead from OhioT1DM-style XML data, with optional LibreView CSV prediction uploads in Streamlit.

This project is educational only. It is not medical advice, diagnostic software, or insulin dosing guidance.

## Project Structure

```text
data/raw/ohio/          # Place OhioT1DM-style XML files here
data/raw/libreview/     # Optional LibreView CSV files
data/processed/         # Notebook outputs
models/                 # Saved trained models and feature columns
notebooks/              # Training and evaluation workflow
src/                    # Reusable preprocessing, feature, training, evaluation, prediction code
streamlit/app.py        # App for loading saved models and making predictions
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name glucose-forecasting
```

## Data Placement

Place OhioT1DM-style XML files anywhere under:

```text
data/raw/ohio/
```

The parser searches recursively for `*.xml` files, so nested patient folders are fine.

Expected XML shape:

```xml
<patient id="559" weight="99" insulin_type="Novalog">
  <glucose_level>
    <event ts="18-01-2022 00:01:00" value="179"/>
    <event ts="18-01-2022 00:06:00" value="183"/>
  </glucose_level>
</patient>
```

LibreView CSV uploads should include:

```text
Timestamp
Historic Glucose mmol/L
Scan Glucose mmol/L
```

The loader also handles common LibreView variants such as metadata rows before the header, `Device Timestamp`, empty scan glucose columns, and glucose values in either mmol/L or mg/dL.

## Notebook Workflow

Run notebooks in order:

1. `notebooks/01_parse_xml_and_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_baseline_models.ipynb`
4. `notebooks/04_xgboost_forecasting.ipynb`
5. `notebooks/05_model_evaluation.ipynb`

Training happens only in notebooks. Streamlit does not train models; it only loads saved model files from `models/`.

## Streamlit

After running the training notebooks and saving models:

```bash
streamlit run streamlit/app.py
```

Upload a LibreView CSV, enter or confirm the current glucose reading, and generate 30, 60, and 120 minute forecasts. The app loads saved model artifacts only; it does not train models.

## Privacy Warning

Glucose traces and timestamps are sensitive health-related data. Keep raw files local, avoid committing data, and remove personal identifiers before sharing outputs. The `.gitignore` excludes raw data, processed data, and saved models by default.

