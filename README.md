# EcoNexus: AI-driven decision support for energy-efficient data centres

Predict -> Recommend -> Simulate -> Compare, on a digital twin of a 6-rack room.
Implements the design in the ABL-1 report (FR1-FR13). Simulated data, advice only.

## Run
    pip install -r requirements.txt
    python scripts/demo.py            # headless end-to-end demo (no Streamlit needed)
    streamlit run src/app.py          # dashboard (first start generates data + trains, ~30 s)
    pytest                            # or: python scripts/run_tests.py
    docker build -t econexus . && docker run -p 8501:8501 econexus

## Layout
    config/    sim.yaml twin.yaml thresholds.yaml   (all parameters, FR12)
    src/data/  workload_loader.py simulator.py      src/twin/  twin.py
    src/ml/    features.py predictor.py             src/decision/  hotspot explainer recommender whatif decision_log
    src/pipeline.py (wiring)  src/app.py (dashboard)  tests/  k8s/  Dockerfile  Jenkinsfile

## Notes
* `data/workload_sample.csv` is a generated trace-like pattern so the project runs offline.
  Replace it with an extract of a public trace (e.g. Google cluster-usage) - any numeric
  `utilisation` column works - and regenerate from Settings.
* The model predicts the temperature *change* over h intervals and adds it to the current value.
