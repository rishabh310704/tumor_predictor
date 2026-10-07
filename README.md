# ML Predictor (Streamlit)

```bash
pip install -r requirements.txt
python train.py          # trains and creates model.joblib
streamlit run app.py     # opens the UI in your browser
```

To use your own data: edit `train.py` (load your CSV, choose the target column)
and update the sliders/inputs in `app.py`.
