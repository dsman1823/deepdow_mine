from setuptools import find_packages, setup

import deepdowmine

DESCRIPTION = "Portfolio optimization with deep learning"
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "cvxpylayers",
    "matplotlib",
    "mlflow",
    "numpy>=1.16.5",
    "pandas",
    "pillow",
    "seaborn",
    "torch>=1.5",
    "tensorboard",
    "tqdm"
]

setup(
    name="deepdowmine",
    version=deepdowmine.__version__,
    author="forked from Jan Krepl project",
    author_email="sei.dmitry.r@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/dsman1823/deepdowmine",
    packages=find_packages(exclude=["tests"]),
    license="Apache License 2.0",
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.6',
    extras_require={
        "dev": ["codecov", "flake8==3.7.9", "pydocstyle", "pytest>=4.6", "pytest-cov", "tox"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "examples": ["sphinx_gallery", "statsmodels"]
    }
)
