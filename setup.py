from setuptools import setup, find_packages

setup(
    name='kidney_mri_registration',
    version='0.1.0',
    description='Motion correction for kidney DCE-MRI via affine + deformable registration',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'numpy>=1.18.0',
        'pydicom>=2.0.0',
        'matplotlib>=3.1.0',
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'tqdm>=4.0.0',
        'scipy>=1.4.0'
    ],
    entry_points={
        'console_scripts': [
            'kidney-mri-train-affine=training.train_affine:main',
            'kidney-mri-train-deformable=training.train_deformable:main',
            'kidney-mri-predict=inference.predict:main',
            'kidney-mri-evaluate=inference.evaluate:main',
            'kidney-mri-run=main:main',
        ],
    },
    python_requires='>=3.7',
)
