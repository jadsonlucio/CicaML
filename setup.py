from setuptools import setup, find_packages

setup(
    name="cicaml",
    version="0.1.2",
    description="CICAML",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=0.23.2",
        "numpy>=1.18.4",
        "pandas>=1.1.0",
        "plotly>=4.14.1",
        "statsmodels>=0.11.1",
        "SQLAlchemy>=1.3.22",
        "nolds>=0.5.2",
        "matplotlib>=3.4.0",
    ],
)
