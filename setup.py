from setuptools import setup, find_packages


setup(
    name="exhaustive-weighted-random-sampler",
    version="0.0.1",
    url="https://github.com/louis-she/exhaustive-weighted-random-sampler",
    author="Chenglu She",
    author_email="chenglu.she@gmail.com",
    description="ExhaustiveWeightedRandomSampler is an advanced version of WeightedRandomSampler",
    packages=find_packages(),
    install_requires=["torch"],
)
