__author__ = 'miky'

# Third-party imports
import numpy
import theano
from theano import tensor
T = tensor
from pylearn2.corruption import Corruptor

class ProteinOneHotCorruptor(Corruptor):
    """
    Corrupts a one-hot vector by changing active element with some
    probability.
    """

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        num_examples = x.shape[0]

        one_hot_size = 21

        N = 11

        keep_mask = T.addbroadcast(
            self.s_rng.binomial(
                size=(num_examples, 1),
                p=1 - self.corruption_level,
                dtype='int8'
            ),
            1
        )

        for i in range(N):

            pvals = T.alloc(1.0 / one_hot_size, one_hot_size)
            if i == 0:
                one_hot = self.s_rng.multinomial(size=(num_examples,), pvals=pvals)
            else:
                one_hot = T.concatenate([one_hot, self.s_rng.multinomial(size=(num_examples,), pvals=pvals) ],axis=1)


        return keep_mask * x + (1 - keep_mask) * one_hot

    def corruption_free_energy(self, corrupted_X, X):
        """
        .. todo::

            WRITEME
        """
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval

