# Contributing

Everyone is welcome to contribute.

There are several ways to contribute to pytransform3d: you could

* send a bug report to the
  [bug tracker](http://github.com/dfki-ric/pytransform3d/issues)
* work on one of the reported issues
* write documentation
* add a new feature
* add tests
* add an example

## How to Contribute Code

This text is shamelessly copied from
[scikit-learn's](https://scikit-learn.org/stable/developers/contributing.html)
contribution guidelines.

The preferred way to contribute to pytransform3d is to fork the
[repository](http://github.com/dfki-ric/pytransform3d/) on GitHub,
then submit a "pull request" (PR):

1. [Create an account](https://github.com/signup/free) on
   GitHub if you do not already have one.

2. Fork the [project repository](http://github.com/dfki-ric/pytransform3d):
   click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.

3. Clone this copy to your local disk:

       $ git clone git@github.com:YourLogin/pytransform3d.git

4. Create a branch to hold your changes:

       $ git checkout -b my-feature

   and start making changes. Never work in the `main` branch!

5. Work on this copy, on your computer, using Git to do the version
   control. When you're done editing, do:

       $ git add modified_files
       $ git commit

   to record your changes in Git, then push them to GitHub with:

       $ git push -u origin my-feature

Finally, go to the web page of the your fork of the pytransform3d repository,
and click 'Pull request' to send your changes to the maintainer for review.
Make sure that your target branch is 'develop'.

In the above setup, your `origin` remote repository points to
YourLogin/pytransform3d.git. If you wish to fetch/merge from the main
repository instead of your forked one, you will need to add another remote
to use instead of `origin`. If we choose the name `upstream` for it, the
command will be:

    $ git remote add upstream https://github.com/dfki-ric/pytransform3d.git

(If any of the above seems like magic to you, then look up the
[Git documentation](http://git-scm.com/documentation) on the web.)

## Requirements for New Features

Adding a new feature to pytransform3d requires a few other changes:

* New classes or functions that are part of the public interface must be
  documented. We use [NumPy's conventions for docstrings](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
* An entry to the API documentation must be added [here](https://dfki-ric.github.io/pytransform3d/api.html).
* Consider writing a simple example script.
* Tests: Unit tests for new features are mandatory. They should cover all
  branches. Exceptions are plotting functions, debug outputs, etc. These
  are usually hard to test and are not a fundamental part of the library.

## Merge Policy

Usually it is not possible to push directly to the develop or main branch for
anyone. Only tiny changes, urgent bugfixes, and maintenance commits can be
pushed directly to the develop branch by the maintainer without a review.
"Tiny" means backwards compatibility is mandatory and all tests must succeed.
No new feature must be added.

Developers have to submit pull requests. Those will be reviewed and merged by
a maintainer. New features must be documented and tested. Breaking changes must
be discussed and announced in advance with deprecation warnings.

## Versioning

Semantic versioning is used, that is, the major version number will be
incremented when the API changes in a backwards incompatible way, the
minor version will be incremented when new functionality is added in a
backwards compatible manner.
