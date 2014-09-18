__author__ = 'miky'
"""
Autoencoders, denoising autoencoders, and stacked DAEs.
"""
# Standard library imports
import functools
from itertools import izip
import operator
from scipy import stats
# Third-party imports
import numpy
import theano
from theano import tensor,shared,function

# Local imports
from pylearn2.blocks import Block, StackedBlocks
from pylearn2.models import Model
from pylearn2.utils import sharedX
from pylearn2.utils.theano_graph import is_pure_elemwise
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.space import VectorSpace
from pylearn2.expr.activations import rescaled_softmax
theano.config.warn.sum_div_dimshuffle_bug = False
import theano.tensor as T

RandomStreams = tensor.shared_randomstreams.RandomStreams

class Autoencoder(Block, Model):
    """
    Base class implementing ordinary autoencoders.

    More exotic variants (denoising, contracting autoencoders) can inherit
    much of the necessary functionality and override what they need.

    Parameters
    ----------
    nvis : int
        Number of visible units (input dimensions) in this model.
        A value of 0 indicates that this block will be left partially
        initialized until later (e.g., when the dataset is loaded and
        its dimensionality is known).  Note: There is currently a bug
        when nvis is set to 0. For now, you should not set nvis to 0.
    nhid : int
        Number of hidden units in this model.
    act_enc : callable or string
        Activation function (elementwise nonlinearity) to use for the
        encoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    act_dec : callable or string
        Activation function (elementwise nonlinearity) to use for the
        decoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
        functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
        for linear units.
    tied_weights : bool, optional
        If `False` (default), a separate set of weights will be allocated
        (and learned) for the encoder and the decoder function. If
        `True`, the decoder weight matrix will be constrained to be equal
        to the transpose of the encoder weight matrix.
    irange : float, optional
        Width of the initial range around 0 from which to sample initial
        values for the weights.
    rng : RandomState object or seed, optional
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):
        super(Autoencoder, self).__init__()
        assert nvis > 0, "Number of visible units must be non-negative"
        assert nhid > 0, "Number of hidden units must be positive"

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(nhid)

        self.N = nvis/21 #numbers of character in a pattern

        # Save a few parameters needed for resizing
        self.nhid = nhid
        self.irange = irange
        self.tied_weights = tied_weights
        self.rng = make_np_rng(rng, which_method="randn")
        self._initialize_hidbias()
        if nvis > 0:
            self._initialize_visbias(nvis)
            self._initialize_weights(nvis)
        else:
            self.visbias = None
            self.weights = None

        seed = int(self.rng.randint(2 ** 30))

        # why a theano rng? should we remove it?
        self.s_rng = make_theano_rng(seed, which_method="uniform")

        if tied_weights and self.weights is not None:
            self.w_prime = self.weights.T
        else:
            self._initialize_w_prime(nvis)

        def _resolve_callable(conf, conf_attr):
            """
            .. todo::

                WRITEME
            """
            if conf[conf_attr] is None or conf[conf_attr] == "linear":
                return None
            # If it's a callable, use it directly.
            if hasattr(conf[conf_attr], '__call__'):
                return conf[conf_attr]
            elif (conf[conf_attr] in globals()
                  and hasattr(globals()[conf[conf_attr]], '__call__')):
                return globals()[conf[conf_attr]]
            elif hasattr(tensor.nnet, conf[conf_attr]):
                return getattr(tensor.nnet, conf[conf_attr])
            elif hasattr(tensor, conf[conf_attr]):
                return getattr(tensor, conf[conf_attr])
            else:
                raise ValueError("Couldn't interpret %s value: '%s'" %
                                    (conf_attr, conf[conf_attr]))

        self.act_enc = _resolve_callable(locals(), 'act_enc')
        self.act_dec = _resolve_callable(locals(), 'act_dec')
        self._params = [
            self.visbias,
            self.hidbias,
            self.weights,
        ]
        if not self.tied_weights:
            self._params.append(self.w_prime)

    def _initialize_weights(self, nvis, rng=None, irange=None):
        """
        .. todo::

            WRITEME
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            (.5 - rng.rand(nvis, self.nhid)) * irange,
            name='W',
            borrow=True
        )

    def _initialize_hidbias(self):
        """
        .. todo::

            WRITEME
        """
        self.hidbias = sharedX(
            numpy.zeros(self.nhid),
            name='hb',
            borrow=True
        )

    def _initialize_visbias(self, nvis):
        """
        .. todo::

            WRITEME
        """
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )

    def _initialize_w_prime(self, nvis, rng=None, irange=None):
        """
        .. todo::

            WRITEME
        """
        assert not self.tied_weights, (
            "Can't initialize w_prime in tied weights model; "
            "this method shouldn't have been called"
        )
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.w_prime = sharedX(
            (.5 - rng.rand(self.nhid, nvis)) * irange,
            name='Wprime',
            borrow=True
        )

    def set_visible_size(self, nvis, rng=None):
        """
        Create and initialize the necessary parameters to accept
        `nvis` sized inputs.

        Parameters
        ----------
        nvis : int
            Number of visible units for the model.
        rng : RandomState object or seed, optional
            NumPy random number generator object (or seed to create one) used \
            to initialize the model parameters. If not provided, the stored \
            rng object (from the time of construction) will be used.
        """
        if self.weights is not None:
            raise ValueError('parameters of this model already initialized; '
                             'create a new object instead')
        if rng is not None:
            self.rng = rng
        else:
            rng = self.rng
        self._initialize_visbias(nvis)
        self._initialize_weights(nvis, rng)
        if not self.tied_weights:
            self._initialize_w_prime(nvis, rng)
        self._set_params()

    def _hidden_activation(self, x):
        """
        Single minibatch activation function.+

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the input minibatch.

        Returns
        -------
        y : tensor_like
            (Symbolic) hidden unit activations given the input.
        """
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

    def _hidden_input(self, x):
        """
        Given a single minibatch, computes the input to the
        activation nonlinearity without applying it.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the input minibatch.

        Returns
        -------
        y : tensor_like
            (Symbolic) input flowing into the hidden layer nonlinearity.
        """
        return self.hidbias + tensor.dot(x, self.weights)

    def upward_pass(self, inputs):
        """
        Wrapper to Autoencoder encode function. Called when autoencoder
        is accessed by mlp.PretrainedLayer

        Parameters
        ----------
        inputs : WRITEME

        Returns
        -------
        WRITEME
        """
        return self.encode(inputs)

    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        encoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after encoding.
        """
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def decode(self, hiddens):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        decoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after decoding.
        """


        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        if isinstance(hiddens, tensor.Variable):
            X = act_dec(self.visbias + tensor.dot(hiddens, self.w_prime))
        else:
            X= [self.decode(v) for v in hiddens]

        g = T.dvector()
        for i in range(self.N):
            start = i * 21
            end = start + 20
            indexis = numpy.arange(start,end+1)
            if i==0:
                g =  rescaled_softmax(theano.tensor.subtensor.take(X,indexis,axis=1,mode='raise'),min_val=1e-5)
            else:
                g =T.concatenate([g,rescaled_softmax(theano.tensor.subtensor.take(X,indexis,axis=1,mode='raise'),min_val=1e-5)],axis=1)

        return g

    def reconstruct(self, inputs):
        """
        Reconstruct (decode) the inputs after mapping through the encoder.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after encoding/decoding.
        """



        return self.decode(self.encode(inputs))


    def __call__(self, inputs):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.

        This just aliases the `encode()` function for syntactic
        sugar/convenience.
        """
        return self.encode(inputs)

    def get_weights(self, borrow=False):
        """
        .. todo::

            WRITEME
        """

        return self.weights.get_value(borrow = borrow)

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """

        return ['v', 'h']

    # Use version defined in Model, rather than Block (which raises
    # NotImplementedError).
    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space


class DenoisingAutoencoder(Autoencoder):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.

    Parameters
    ----------
    corruptor : object
        Instance of a corruptor object to use for corrupting the
        input.
    nvis : int
        WRITEME
    nhid : int
        WRITEME
    act_enc : WRITEME
    act_dec : WRITEME
    tied_weights : bool, optional
        WRITEME
    irange : WRITEME
    rng : WRITEME

    Notes
    -----
    The remaining parameters are identical to those of the constructor
    for the Autoencoder class; see the `Autoencoder.__init__` docstring
    for details.
    """
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):
        super(DenoisingAutoencoder, self).__init__(
            nvis,
            nhid,
            act_enc,
            act_dec,
            tied_weights,
            irange,
            rng
        )
        self.corruptor = corruptor

    def reconstruct(self, inputs):
        """
        Reconstruct the inputs after corrupting and mapping through the
        encoder and decoder.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be corrupted and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after corruption and encoding/decoding.
        """
        corrupted = self.corruptor(inputs)

        return super(DenoisingAutoencoder, self).reconstruct(corrupted)


