"""SonarSense backend: FastAPI service wiring together ingestion ->
preprocessing -> YOLO detection -> VAE anomaly analysis -> confidence
scoring -> geolocation -> SQLite persistence, behind a REST + WebSocket API
a frontend can attach to.
"""
