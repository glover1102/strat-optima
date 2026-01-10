"""
Setup configuration for QTAlgo Super26 Walk-Forward Optimization Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="strat-optima",
    version="1.0.0",
    author="QTAlgo Team",
    author_email="contact@qtalgo.com",
    description="Walk-Forward Optimization Framework for QTAlgo Super26 Trading Strategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glover1102/strat-optima",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "TA-Lib>=0.4.28",
        "pandas-ta>=0.3.14b",
        "scikit-learn>=1.3.0",
        "optuna>=3.3.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "yfinance>=0.2.28",
        "ccxt>=4.1.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.5.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "strat-optima=main:cli_main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
