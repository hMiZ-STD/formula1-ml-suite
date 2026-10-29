"""Pytest shared configuration and global environment hooks."""
import os

# Suppress joblib loky physical core detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
