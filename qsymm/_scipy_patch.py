# This file has been taken from Scipy 1.1.0 under the
# BSD 3-Clause license, reproduced below:
#
# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pkg_resources import parse_version

import scipy.sparse

# Scipy 1.1 implemented sparse matrix reshaping. In order to allow a lower
# version of scipy (to remain compatible with Kwant's requirements) we do
# not want to use this more recent version of scipy. We monkey patch the
# "reshape" method onto the appropriate sparse matrix classes.
#
# This module should be imported in all modules that use the "reshape"
# method of sparse matrices (this is a sufficient but not necessary
# condition for the monkey patching to work, because the patch may already
# have been applied by other modules that themselves import this).
#
# TODO: remove this file (and all the places it is imported) when we can
#       depend on scipy >= 1.1

if parse_version(scipy.__version__) < parse_version("1.1"):

    import operator

    import numpy as np
    from scipy.sparse import coo_matrix


    def check_shape(args, current_shape=None):
        """Imitate numpy.matrix handling of shape arguments"""
        if len(args) == 0:
            raise TypeError("function missing 1 required positional argument: "
                            "'shape'")
        elif len(args) == 1:
            try:
                shape_iter = iter(args[0])
            except TypeError:
                new_shape = (operator.index(args[0]), )
            else:
                new_shape = tuple(operator.index(arg) for arg in shape_iter)
        else:
            new_shape = tuple(operator.index(arg) for arg in args)

        if current_shape is None:
            if len(new_shape) != 2:
                raise ValueError('shape must be a 2-tuple of positive integers')
            elif new_shape[0] < 0 or new_shape[1] < 0:
                raise ValueError("'shape' elements cannot be negative")

        else:
            # Check the current size only if needed
            current_size = np.prod(current_shape, dtype=int)

            # Check for negatives
            negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
            if len(negative_indexes) == 0:
                new_size = np.prod(new_shape, dtype=int)
                if new_size != current_size:
                    raise ValueError('cannot reshape array of size {} into shape {}'
                                     .format(current_size, new_shape))
            elif len(negative_indexes) == 1:
                skip = negative_indexes[0]
                specified = np.prod(new_shape[0:skip] + new_shape[skip+1:])
                unspecified, remainder = divmod(current_size, specified)
                if remainder != 0:
                    err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                    raise ValueError('cannot reshape array of size {} into shape {}'
                                     ''.format(current_size, err_shape))
                new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip+1:]
            else:
                raise ValueError('can only specify one unknown dimension')

            # Add and remove ones like numpy.matrix.reshape
            if len(new_shape) != 2:
                new_shape = tuple(arg for arg in new_shape if arg != 1)

                if len(new_shape) == 0:
                    new_shape = (1, 1)
                elif len(new_shape) == 1:
                    new_shape = (1, new_shape[0])

        if len(new_shape) > 2:
            raise ValueError('shape too large to be a matrix')

        return new_shape


    def check_reshape_kwargs(kwargs):
        """Unpack keyword arguments for reshape function.
        This is useful because keyword arguments after star arguments are not
        allowed in Python 2, but star keyword arguments are. This function unpacks
        'order' and 'copy' from the star keyword arguments (with defaults) and
        throws an error for any remaining.
        """

        order = kwargs.pop('order', 'C')
        copy = kwargs.pop('copy', False)
        if kwargs:  # Some unused kwargs remain
            raise TypeError('reshape() got unexpected keywords arguments: {}'
                            .format(', '.join(kwargs.keys())))
        return order, copy


    def get_index_dtype(arrays=(), maxval=None, check_contents=False):

        int32min = np.iinfo(np.int32).min
        int32max = np.iinfo(np.int32).max

        dtype = np.intc
        if maxval is not None:
            if maxval > int32max:
                dtype = np.int64

        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)

        for arr in arrays:
            arr = np.asarray(arr)
            if not np.can_cast(arr.dtype, np.int32):
                if check_contents:
                    if arr.size == 0:
                        # a bigger type not needed
                        continue
                    elif np.issubdtype(arr.dtype, np.integer):
                        maxval = arr.max()
                        minval = arr.min()
                        if minval >= int32min and maxval <= int32max:
                            # a bigger type not needed
                            continue

                dtype = np.int64
                break

        return dtype


    def sparse_reshape(self, *args, **kwargs):
        # If the shape already matches, don't bother doing an actual reshape
        # Otherwise, the default is to convert to COO and use its reshape
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)


    def coo_reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        nrows, ncols = self.shape

        if order == 'C':
            # Upcast to avoid overflows: the coo_matrix constructor
            # below will downcast the results to a smaller dtype, if
            # possible.
            dtype = get_index_dtype(maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))

            flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            dtype = get_index_dtype(maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))

            flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of new_row and new_col regardless
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return coo_matrix((new_data, (new_row, new_col)),
                          shape=shape, copy=False)


    # Apply monkey patches
    scipy.sparse.spmatrix.reshape = sparse_reshape
    scipy.sparse.coo_matrix.reshape = coo_reshape
