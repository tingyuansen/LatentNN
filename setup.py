"""
LatentNN: Correcting Attenuation Bias in Neural Networks

Installation:
    pip install -e .

Or for development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="latentnn",
    version="1.0.0",
    author="Yuan-Sen Ting",
    author_email="ting.74@osu.edu",
    description="Correcting attenuation bias in neural networks by treating inputs as latent variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tingyuansen/LatentNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.5.0",
            "tqdm>=4.60.0",
            "jupyter>=1.0.0",
            "joblib>=1.1.0",
        ],
    },
)

