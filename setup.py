from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='quati',
    version='0.0.1',
    description='Simple library for doc. classification and seq. tagging',
    long_description=readme,
    author='Marcos Treviso',
    author_email='marcosvtreviso@gmail.com',
    url='https://github.com/deep-spin/quati',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    data_files=['LICENSE'],
    zip_safe=False,
    keywords='evaluator',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
