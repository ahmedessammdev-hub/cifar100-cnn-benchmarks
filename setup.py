from setuptools import setup, find_packages

setup(
    name="cnn-benchmark-lab",
    version="1.0.0",
    description="Comparing Custom CNN vs Transfer Learning on CIFAR-100",
    author="Ahmed",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dashboard": ["streamlit>=1.28.0"],
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "python-multipart>=0.0.6"],
        "tracking": ["wandb>=0.16.0"],
        "all": [
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "python-multipart>=0.0.6",
            "wandb>=0.16.0",
        ],
    },
)
