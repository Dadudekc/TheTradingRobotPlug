from setuptools import setup, find_packages
import os

# Fallback for long_description if README.md is missing
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="TradingRobotPlug2",
    version="1.0.0",
    description="A utility project for fetching and processing financial data from APIs and news sources.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Victor Dixon",
    author_email="DaDudeKC@gmail.com",
    url="https://github.com/Dadudekc/TheTradingRobotPlug",
    packages=find_packages(where="src", exclude=["tests*", "docs*"]),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "aiohttp>=3.8.1",
        "alpaca-trade-api>=2.3.0",
        "asyncio>=3.4.3",
        "python-dotenv>=0.21.0",
        "pandas>=1.4.3",
        "pytest>=7.2.0",
        "pytest-asyncio>=0.20.1",
        "aioresponses>=0.7.2",
        "requests>=2.28.1",
        "textblob>=0.15.3",
        "yfinance>=0.2.12",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "flake8",
            "black",
            "mypy",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-trading-bot=main:run",
        ],
    },
    keywords=["stock trading", "financial data", "utilities", "APIs"],
    project_urls={
        "Bug Tracker": "https://github.com/Dadudekc/TheTradingRobotPlug/issues",
        "Source Code": "https://github.com/Dadudekc/TheTradingRobotPlug",
    },
)
