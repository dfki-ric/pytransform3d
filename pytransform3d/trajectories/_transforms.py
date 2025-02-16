import numpy as np


def invert_transforms(A2Bs):
    """Invert transforms.

    Parameters
    ----------
    A2Bs : array-like, shape (..., 4, 4)
        Transforms from frames A to frames B

    Returns
    -------
    B2As : array, shape (..., 4, 4)
        Transforms from frames B to frames A

    See Also
    --------
    pytransform3d.transformations.invert_transform :
        Invert one transformation.
    """
    A2Bs = np.asarray(A2Bs)
    instances_shape = A2Bs.shape[:-2]
    B2As = np.empty_like(A2Bs)
    # ( R t )^-1   ( R^T -R^T*t )
    # ( 0 1 )    = ( 0    1     )
    B2As[..., :3, :3] = A2Bs[..., :3, :3].transpose(
        tuple(range(A2Bs.ndim - 2)) + (A2Bs.ndim - 1, A2Bs.ndim - 2)
    )
    B2As[..., :3, 3] = np.einsum(
        "nij,nj->ni",
        -B2As[..., :3, :3].reshape(-1, 3, 3),
        A2Bs[..., :3, 3].reshape(-1, 3),
    ).reshape(*(instances_shape + (3,)))
    B2As[..., 3, :3] = 0.0
    B2As[..., 3, 3] = 1.0
    return B2As


def concat_one_to_many(A2B, B2Cs):
    """Concatenate transformation A2B with multiple transformations B2C.

    We use the extrinsic convention, which means that B2Cs are left-multiplied
    to A2B.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    B2Cs : array-like, shape (n_transforms, 4, 4)
        Transforms from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate multiple transformations with one.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("nij,jk->nik", B2Cs, A2B)


def concat_many_to_one(A2Bs, B2C):
    """Concatenate multiple transformations A2B with transformation B2C.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs.

    Parameters
    ----------
    A2Bs : array-like, shape (n_transforms, 4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_one_to_many :
        Concatenate one transformation with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("ij,njk->nik", B2C, A2Bs)


def concat_many_to_many(A2B, B2C):
    """Concatenate multiple transformations A2B with transformation B2C.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs.

    Parameters
    ----------
    A2B : array-like, shape (n_transforms, 4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (n_transforms, 4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate one transformation with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("ijk,ikl->ijl", B2C, A2B)


def concat_dynamic(A2B, B2C):
    """Concatenate multiple transformations A2B,B2C with different shapes.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs. it can handle different shapes of A2B and B2C dynamically.

    Parameters
    ----------
    A2B : array-like, shape (n_transforms, 4, 4) or (4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (n_transforms, 4, 4) or (4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4) or (4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate multiple transformations with one transformation.

    concat_one_to_many :
        Concatenate one transformation with multiple transformations.

    concat_many_to_many :
        Concatenate multiple transformations with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    A2B = np.asarray(A2B)
    B2C = np.asarray(B2C)
    if B2C.ndim == 2 and A2B.ndim == 2:
        return B2C.dot(A2B)
    elif B2C.ndim == 2 and A2B.ndim == 3:
        return concat_many_to_one(A2B, B2C)
    elif B2C.ndim == 3 and A2B.ndim == 2:
        return concat_one_to_many(A2B, B2C)
    elif B2C.ndim == 3 and A2B.ndim == 3:
        return concat_many_to_many(A2B, B2C)
    else:
        raise ValueError(
            "Expected ndim 2 or 3; got B2C.ndim=%d, A2B.ndim=%d"
            % (B2C.ndim, A2B.ndim)
        )
