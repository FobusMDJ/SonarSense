"""VAE-based anomaly detection stage: trains a convolutional VAE on
"hard negative" seafloor background patches (real seafloor, no known
marine debris) so reconstruction error at inference flags real objects as
anomalies, downstream of the preprocessing pipeline's denoised output.
"""
