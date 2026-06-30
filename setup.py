from setuptools import setup, find_packages

setup(
    name="spqr-iomt",
    version="1.0.0",
    description="Secure & Privacy-Preserving Quantum-Resilient IoMT Platform",
    author="SPQR-IoMT Research Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "opacus>=1.4.0",
            "flwr>=1.5.0",
            "tenseal>=0.3.14",
            "liboqs-python>=0.8.0",
            "cryptography>=41.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "spqr-benchmark=experiments.run_all_experiments:main",
            "spqr-server=federated_learning.fl_server:main",
            "spqr-client=federated_learning.fl_client:main",
            "spqr-gateway=hybrid_migration.gateway:main",
        ]
    },
)
