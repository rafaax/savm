from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="savm",
    version="0.1.0",
    description="Sistema de Análise e Vetorização de Módulos para detecção de SQL Injection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaax/savm",
    author="Raphael Meireles",
    author_email="raphael.meireles@ssector7.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="security, sql injection, machine learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    install_requires=[
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "matplotlib>=3.4.0",
        "wordcloud>=1.8.1",
    ],
    package_data={
        "savm": ["data/*.csv", "config/*.json"],
    },
    project_urls={ 
        "Bug Reports": "https://github.com/rafaax/savm/issues",
        "Source": "https://github.com/rafaax/savm",
    },
)