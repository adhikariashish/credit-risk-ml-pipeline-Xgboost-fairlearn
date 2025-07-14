# setup.py
from setuptools import setup, find_packages

setup(
    name="credit_risk_ml_pipeline_Xgboost+fairlearn",
    version="0.1.0",
    packages=find_packages(include=["pipeline_config", "pipeline_config.*"]),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "pipeline_config": ["config.yaml"],
    },
    install_requires=[
        "pyyaml",
        "xgboost",
        "fairlearn",
    ],
)
