import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'tes_simulator',
    version = '0.1',
    author = 'Tommaso Ghigna',
    author_email = 't.ghigna@gmail.com',
    description = 'Simulation tool for DC and AC biased TES detectors',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/tomma90/tes_simulator',
    packages=setuptools.find_packages()
)
