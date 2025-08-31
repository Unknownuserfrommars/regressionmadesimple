"""
Setup script for RegressionMadeSimple package.
"""

from setuptools import setup, find_packages

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A simple wrapper around scikit-learn for regression tasks."

# Read requirements
install_requires = [
    "numpy>=1.19.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
]

setup(
    name="regressionmadesimple",
    version="2.0.0",
    description="A simple wrapper around scikit-learn for regression tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unknownuserfrommars/regressionmadesimple",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    keywords="regression, machine learning, scikit-learn, wrapper, simple",
    project_urls={
        "Bug Reports": "https://github.com/Unknownuserfrommars/regressionmadesimple/issues",
        "Source": "https://github.com/Unknownuserfrommars/regressionmadesimple",
        "Documentation": "https://unknownuserfrommars.github.io/regressionmadesimple/",
    },
)