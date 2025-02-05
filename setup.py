from setuptools import setup, find_packages

setup(
    name="cddd-onnx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "onnxruntime",
        "requests",
        "tqdm",
        "numpy",
        "pandas",
        "rdkit",
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ]
    },
    author="Sergey Sosnin",
    author_email="sergey.sosnin@univie.ac.at",
    description="CDDD models in ONNX format with automatic model downloading",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sergsb/cddd-onnx",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'cddd-onnx = cddd_onnx.__main__:run',
        ],
    },
)
