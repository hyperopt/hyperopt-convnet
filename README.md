Hyperparameter Optimization for Convolutional Vision Architectures
==================================================================

This package provides a [Theano](http://www.deeplearning.net/software/theano)-based implementation of convolutional networks
as described in (Bergstra, Yamins, and Cox, 2013), which exposes many
architectural hyperparameters for optimization by
[hyperopt](http://jaberg.github.com/hyperopt).

# Installation

1. Requirements:


   * A Python/Numpy/Scipy stack.
     The Python package requirements are listed in
     requirements.txt.

   * Optional (but strongly recommended) is an NVIDIA GPU device at least as
     fast as, say, a GTX280, and CUDA. See Theano's documentation for setting
     up Theano to use a GPU device.

   * Optional (but strongly recommended) is the MongoDB database software,
     which allows hyperopt to support parallel optimization.

2. Check out this project

   `git clone https://github.com/jaberg/hyperopt-convnet.git`.

3. Install it as a Python package. This installation makes the code files
   importable, which is required when running asynchronous hyperparameter
   optimization (i.e. with hyperopt-mongo-worker, as explained below).

   `python setup.py install`

   Consider installing this within your user account (`--user`) or within a
   virtualenv to avoid installing this package system-wide, and to avoid
   needing root privileges.


# Testing

If installation goes well, then you will now be able to import the `hpconvnet`
module.

The quickest smoke test that things are set up properly is to just run a
small-scale simulation on the cifar10 data set:

`
  THEANO_FLAGS=device=gpu shovel cifar10.small_random_run
`

A few larger end-to-end regression tests can be run from the
`hpconvnet/tests/` folder.


# Running An Experiment in Parallel with MongoDB

Running hyperparameter optimization on large convolutional networks for data
sets such as [LFW](http://vis-www.cs.umass.edu/lfw/)
and [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) takes a significant amount of time:
expect it to take about a GPU-week.
This cannot be completely parallelized (Bayesian optimization works on the
basis of feedback about the fitness landscape after all), but in my experience
it can easily be parallelized 5-fold to 10-fold.
So if you have access to a small cluster you can see significant progress in
an hour or two, and be done in a day.

What follows here is a sketch of the unix commands you would need to do to
make this happen.
To get more of a sense about what's going on, read through
[hyperopt documentation on using
mongo](https://github.com/jaberg/hyperopt/wiki/Parallelizing-search).


1. Set up a mongodb process for inter-process communication.

   `
    mongod --dbpath . --port PORT --directoryperdb --fork --journal --logpath log.log --nohttpinterface
   `

    If this machine is visible to the internet, you should either bind mongod
    to the local loopback address and connect to the database via an ssh
    tunnel, or set up mongodb for password-protected access.

2. Start an asynchronous search process, that connects to the mongodb and
   polls a work queue created there.

   `
    shovel cifar10.random localhost PORT
   `

3. Start one or more generic hyperopt worker processes to crank through the
   trials of the experiment.

   `
   ssh WORKNODE hyperopt-mongo-worker --mongo=localhost:PORT/cifar_db1
   `

   If you have a cluster with a queue system (e.g. Torque, PBS, etc.) then use
   that system to schedule a few hyperopt-mongo-worker processes. When they
   start, they will connect to the database and reserve an experiment trial.
   These processes will loop indefinitely by default, dequeueing/reserving trials
   and storing the results back to the database. They will stop when the
   search process no longer adds new trials to the database, or when several
   (4) consecutive trials fail to complete successfully (i.e. your trial
   evaluation code is faulty and you should either fix it or at least catch the
   terminating exceptions).



# References

* J. Bergstra, D. Yamins, D. D. Cox (2013).
  [Making a Science of Model Search](forthcoming),
  in Proc. ICML. -- This paper describes the convolutional architectures
  implemented in this software package, and the results you should expect from
  hyperparameter optimization.

* J. Bergstra, R. Bardenet, Y. Bengio, B. Kegl (2011).
  [Algorithms for Hyper-parameter Optimization](http://books.nips.cc/papers/files/nips24/NIPS2011_1385.pdf)
  In Proc. NIPS. -- This paper introduces the TPE hyperparameter optimization algorithm.
