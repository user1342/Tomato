from setuptools import setup, find_packages

setup(
    name="Tomato",  # Name of the tool/package
    version="0.1.9",  # Initial version
    url="https://github.com/user1342/Tomato",  # Replace with the project's GitHub URL
    packages=find_packages(),  # Automatically find package directories
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0",
        "Operating System :: Ubuntu",
    ],
    python_requires=">=3.8",  # Minimum Python version required
    entry_points={
        'console_scripts': [
            'tomato-encode=tomato.cli:main',  # maps the command `tomato-encode` to the `main` function in `cli.py`
            'tomato-decode=tomato.cli:main',  # maps the command `tomato-decode` to the `main` function in `cli.py`
        ],
    },
)
