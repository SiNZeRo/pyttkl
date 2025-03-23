from setuptools import setup, find_packages

setup(
    name="pyttkl",
    version="0.1.0",
    description="Some Python tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Truman TIAN",
    author_email="flametbs@gmail.com",
    url="https://github.com/SiNZeRo/pyttkl",  # Replace with the actual URL
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "zstandard",
        "lz4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
