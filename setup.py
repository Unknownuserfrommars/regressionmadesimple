from setuptools import setup, find_packages

setup(
    name="regressionmadesimple",
    version="1.1.1",
    description="Minimalist ML toolkit wrapping sklearn for quick prototyping. Just `import rms` and go.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Unknownuserfrommars",
    # author_email="your_email@example.com",  # optional, can leave blank
    # url="https://github.com/Unknownuserfrommars/regressionmadesimple",  # optional because not created yet
    license="MIT",
    python_requires=">=3.10",

    packages=find_packages(),  # Automatically finds all modules in the folder
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
    ],
    include_package_data=True,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning regression sklearn wrapper linear",
    # project_urls={
    #     "Bug Tracker": "https://github.com/yourusername/regressionmadesimple/issues",
    #     "Documentation": "https://github.com/yourusername/regressionmadesimple#readme",
    # },
)
