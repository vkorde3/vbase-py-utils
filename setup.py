"""
validityBase Python Utilities
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="vbase-utils",
    version="0.0.1",
    author="validityBase",
    author_email="tech@vbase.com",
    description="validityBase Python Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/validityBase/vbase-py-utils",
    packages=find_packages(),
    package_data={
        "": ["../requirements.txt"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
