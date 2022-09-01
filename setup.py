from setuptools import setup, find_packages

setup(
    name        = 'custom_network_layer',
    version     = '0.1.0',
    description = 'pytorch custom network',
    packages    = find_packages(where='custom_network_layer'),
    package_dir = {'': 'custom_network_layer'},
)