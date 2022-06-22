import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fp:
    install_requires = fp.read()

setuptools.setup(
    name="normalization",
    version="1.0.0",
    description="Medical Concepts Normalization",
    long_description=long_description,
    long_description_conttype="text/markdown",
    packages=setuptools.find_packages(exclude=["notebooks"]),
    install_requires=install_requires,
    zip_safe=False,
)