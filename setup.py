# coding: UTF-8
from setuptools import setup, find_packages

setup(
    name="wsss",
    version="0.2.0",
    description="Weakly Supervised Semantic Segmentation implementations using tensorflow.",
    long_description="See the README.md at https://github.com/kthfan/Weakly-Supervised-Semantic-Segmentation",
    author="kthfan",
    author_email="3999932@gmail.com",
    url="https://github.com/kthfan/Weakly-Supervised-Semantic-Segmentation",
    packages=find_packages(
        where='.',
        include=['wsss/*'],
    ),
    # package_dir={"": "wsss"}
)



