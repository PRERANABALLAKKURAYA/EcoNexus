FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY config/ config/
COPY .streamlit/ .streamlit/
COPY data/workload_sample.csv data/workload_sample.csv
# Generate the simulated telemetry, train the model and pre-compute the live-twin
# playback at build time, so the container starts instantly and every pod is identical.
RUN python -c "from src.pipeline import build_system; from src.dashboard.playback import load_frames; load_frames(build_system('.'))"
RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER 10001
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
