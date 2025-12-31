from setuptools import setup, find_packages

setup(
    name="regressionmadesimple",
    version="3.0.0",
    description="Minimalist machine learning toolkit that wraps `scikit-learn` for quick prototyping. Just `import rms` and go.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Unknownuserfrommars",
    # author_email="your_email@example.com",  # optional, can leave blank
    url="https://github.com/Unknownuserfrommars/regressionmadesimple",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
        "matplotlib",
        "joblib",  # Added for model serialization in v3.0.0
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
    ],
    keywords="machine-learning regression sklearn wrapper polynomial curve-fitting",
)