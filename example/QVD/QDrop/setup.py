from setuptools import setup,find_packages

setup(
    name="QDrop",
    version='0.1dev',
    packages=find_packages(),
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
