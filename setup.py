from setuptools import setup, find_packages
from ocr2txt import VERSION

setup(
    name="ocr2txt",
    packages=find_packages(),
    version=VERSION,
    install_requires=[
        'torch==1.7',
        'torchvision==0.8.1',
        'transformers==3.5.1',
        'pytorch-lightning==1.0.6',
        'datasets==1.1.2',
        'pillow',
        'spacy',
    ]
)
