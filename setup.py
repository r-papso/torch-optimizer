import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    long_description = long_description.replace(
        "](./", "](https://github.com/r-papso/torch-optimizer/blob/main/"
    )


setuptools.setup(
    name="torch-opt",
    version="0.0.1",
    author="Rastislav Papso",
    author_email="rastislav.papso@gmail.com",
    description="PyTorch models optimization using neural network pruning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/r-papso/torch-optimizer",
    project_urls={"Bug Tracker": "https://github.com/r-papso/torch-optimizer/issues",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["torchopt", "torchopt.model", "torchopt.optim", "torchopt.prune", "torchopt.train",],
    install_requires=[
        "deap>=1.3.1",
        "pytorch-ignite>=0.4.8",
        "thop>=0.0.31",
        "torch>=1.10.0",
        "torch-pruning>=0.2.7",
        "torchvision>=0.11.1",
    ],
    python_requires=">=3.9",
)
