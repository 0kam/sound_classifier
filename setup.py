from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

with open('LICENSE') as f:
    license = f.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="sound-classifier",
    version="0.1.0",
    license=license,
    description="Training and deploying deep sound classifiers, especially for bio-acoustic monitorings.",
    author="Ryotaro Okamoto",
    url="https://github.com/0kam/sound_classifier",
    packages=find_packages("sound_classifier"),
    package_dir={"": "sound_classifier"},
    py_modules=[splitext(basename(path))[0] for path in glob('sound_classifier/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)