# pytransform

<a href="https://git.hb.dfki.de/ci/projects/1?ref=master">
<img src="https://git.hb.dfki.de/ci/projects/1/status.png?ref=master" /></a>

A Python library for transformations in three dimensions. The library focuses
on readability and debugging, not on computational efficiency. If you want to
have an efficient implementation of some function from the library you can
easily extract the relevant code and implement it more efficiently in a
language of your choice.

pytransform relies on NumPy for linear algebra and on Matplotlib to offer
plotting functionalities. Some modules require SciPy for more complicated
algorithms.

## Installation

Install the package with:

    python setup.py install

## Documentation

The docmentation of this project can be found in the directory `doc`. To
build the documentation, run e.g. (on unix):

    cd doc
    make html

The HTML documentation is now located at `doc/build/html/index.html`.

## Tests

You can use nosetests to run the tests of this project in the root directory:

    nosetests

A coverage report will be located at `cover/index.html`.
