from setuptools import setup, find_packages


setup(
    name="exhaustive-weighted-random-sampler",
    version="0.0.2",
    url="https://github.com/louis-she/exhaustive-weighted-random-sampler",
    author="Chenglu She",
    author_email="chenglu.she@gmail.com",
    description="ExhaustiveWeightedRandomSampler is an advanced version of WeightedRandomSampler",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=["torch"],
)
