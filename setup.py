from setuptools import setup, find_packages

# Read requirements from file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="GSDB_classification",
    version="0.1.0",
    description="Gastric Slide Database - Classification Pipeline",
    author="Dominic Liebel",
    author_email="dominic.liebel@gmail.com",
    url="https://github.com/DominicLiebel/GSDB",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)