import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extras_requirements = {
    "dev": ["wheel", "black", "pytest", "mypy"],
}

setuptools.setup(
    name="terms",
    version="0.1.0",
    author="hp0404",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        "terms": ["assets/*.json"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "spacy>=3.4.0",
        "pandas>=1.0.0",
    ],
    extras_require=extras_requirements,
    python_requires=">=3.7",
)
