"""
CyberSecLLM: Hybrid Mamba-Transformer Foundation Model
for Zero-Shot Cybersecurity Threat Intelligence
with Continuous-Time Adaptive Detection

Author: Roger Nick Anaedevha
Affiliation: National Research Nuclear University MEPhI
"""

from setuptools import setup, find_packages

setup(
    name="cybersecllm",
    version="1.0.0",
    author="Roger Nick Anaedevha",
    author_email="ar006@campus.mephi.ru",
    description=(
        "A hybrid Mamba-Transformer foundation model with TA-BN-ODE "
        "and temporal point processes for zero-shot cybersecurity "
        "threat intelligence"
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/rogerpanel/CyberSecLLM-models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchdiffeq>=0.2.3",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
