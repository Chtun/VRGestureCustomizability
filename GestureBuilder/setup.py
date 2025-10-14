"""Package configuration and setup for improved GestureBuilder implementation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Package requirements
REQUIREMENTS = [
    "torch",
    "sklearn",
    "pandas",
    "numpy",
    "plotly",
]

# Development requirements
DEV_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0",
    "isort>=5.0.0",
    "flake8>=3.9.0",
    "mypy>=0.900",
]

# Documentation requirements
DOCS_REQUIREMENTS = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "myst-parser>=0.15.0",
]

setup(
    name="GestureBuilder",
    version="0.1.0",
    description="A re-implementation of Gesture Builder, with additional functionality of two-hand gesture recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chitsein Htun",
    author_email="chtun@live.com",
    url="https://github.com/Chtun/VRGestureCustomizability",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Package dependencies
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "docs": DOCS_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + DOCS_REQUIREMENTS,
    },

    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",

    # Package data
    include_package_data=True,
    package_data={
        "GestureBuilder": [
            "data/test_data/*.csv",
            "data/example_data/*.csv",
        ]
    },

    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "GestureBuilder=GestureBuilder.cli:main",
        ]
    },

    # Additional metadata
    platforms=["any"],
    license="MIT",
)
