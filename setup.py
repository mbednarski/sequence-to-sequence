from setuptools import find_packages, setup

print(find_packages(where="src"))

setup(name="seq2seq", packages=find_packages(where="src"), package_dir={"": "src"})
