from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_requirements(path: str) -> List[str]:
    """Collect requirement specifiers from a pip-compatible requirements file."""
    requirements: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#") and entry != "-e .":
            requirements.append(entry)
    return requirements


def read_text(path: str) -> str:
    """Read a UTF-8 text file and fall back to an empty string if it is missing."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8") if file_path.exists() else ""


setup(
    name="grade-prediction",
    version="0.0.1",
    author="Can Sonmez",
    author_email="cansonmez06@outlook.com.tr",
    description="Tools for predicting student grades with machine learning.",
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/cansonmez/GradePrediciton",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/cansonmez/GradePrediciton/issues",
    },
)
