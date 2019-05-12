import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Basic Machine Learning Stock Prediction",
    version="0.0.1",
    author="Leandro Izquierdo",
    author_email="leandro.izqvallejos@gmail.com",
    description="Package that gathers data from the Yahoo API and makes predictions using a Random Forest alorithm",
    long_description=long_description,
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: >3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)