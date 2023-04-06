from glob import glob

from setuptools import setup
from pathlib import Path

with open('LICENSE') as f:
    license = f.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="soundclassifier",
    version="0.1.0",
    license=license,
    description="Training and deploying deep sound classifiers, especially for bio-acoustic monitorings.",
    author="Ryotaro Okamoto",
    url="https://github.com/0kam/sound_classifier",
    packages=["soundclassifier", "soundclassifier.core", "soundclassifier.models"],
    package_dir={"soundclassifier": "soundclassifier", "soundclassifier.core": "soundclassifier/core", "soundclassifier.models": "soundclassifier/models"},
    py_modules=[(str(Path(path).parent) + "/" + Path(path).stem).replace("/", ".") for path in glob('soundclassifier/**/*.py')],
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)