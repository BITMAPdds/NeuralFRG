from setuptools import setup, find_namespace_packages

setup(
    name = 'neuralfrg',
    version = '0.1.0',   
    description = "Parameterized Neural ODEs on fRG data",
    long_description="""A PyTorch implementation of
        Parameterized Neural Ordinary Differential Equations (PNODE)
        learning of functional Renormalization Group (fRG) models""",
    url = 'https://github.com/Matematija/NeuralFRG',
    author = 'Matija Medvidovic',
    author_email = 'matija.medvidovic@columbia.edu',
    license = 'Apache License Version 2.0',
    packages = find_namespace_packages(),
    install_requires = [
        'psutil',
        'numpy',
        'scipy',
        'h5py',
        'torch>=1.10',
        'torchvision>=0.11',
        'torchdiffeq>=0.2'
    ],
    classifiers = [
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Environment :: GPU :: NVIDIA CUDA :: 11.4',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)