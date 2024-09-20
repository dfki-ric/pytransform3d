==============================
Uncertainty in Transformations
==============================

In computer vision and robotics we are never absolutely certain about
transformations. It is often a good idea to express the uncertainty explicitly
and use it. The module :mod:`pytransform3d.uncertainty` offers tools to handle
with uncertain transformations.

----------------------------------------
Gaussian Distribution of Transformations
----------------------------------------

Uncertain transformations are often represented by Gaussian distributions,
where the mean of the distribution is represented by a transformation matrix
:math:`\boldsymbol{T} \in SE(3)` and the covariance is defined in the tangent space
through exponential coordinates
:math:`\boldsymbol{\Sigma} \in \mathbb{R}^{6 \times 6}`.

.. warning::

    It makes a difference whether the uncertainty is defined in the global
    frame of reference (i.e., left-multiplied) or local frame of reference
    (i.e., right-multiplied). Unless otherwise stated, we define uncertainty
    in a global frame, that is, to sample from a Gaussian distribution of
    transformations
    :math:`\boldsymbol{T}_{xA} \sim \mathcal{N}(\boldsymbol{T}|\boldsymbol{T}_{BA}, \boldsymbol{\Sigma}_{6 \times 6}))`,
    we compute :math:`\Delta \boldsymbol{T}_{xB} \boldsymbol{T}_{BA}`,
    with :math:`\Delta \boldsymbol{T}_{xB} = Exp(\boldsymbol{\xi})` and
    :math:`\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}_6, \boldsymbol{\Sigma}_{6 \times 6})`.
    Hence, the uncertainty is defined in the global frame B, not in the local
    body frame A.

We can use
:func:`~pytransform3d.uncertainty.estimate_gaussian_transform_from_samples`
to estimate a Gaussian distribution of transformations. We can sample from
a known Gaussian distribution of transformations with
:func:`~pytransform3d.transformations.random_transform` as illustrated in
the Example :ref:`sphx_glr__auto_examples_plots_plot_sample_transforms.py`.

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_sample_transforms_001.png
   :target: ../_auto_examples/plots/plot_sample_transforms.html
   :align: center

----------------------------
Visualization of Uncertainty
----------------------------

A typical visual representation of Gaussian distributions in 3D coordinates
are equiprobable ellipsoids (obtained with
:func:`~pytransform3d.uncertainty.to_ellipsoid`). This is equivalent to showing
the :math:`k\sigma, k \in \mathbb{R}` intervals of a 1D Gaussian distribution.
However, for transformations are also interactions between rotation and
translation components so that an ellipsoid is not an appropriate
representation to visualize the distribution of transformations in 3D. We have
to project a 6D hyper-ellipsoid to 3D (for which we can use
:func:`~pytransform3d.uncertainty.to_projected_ellipsoid`), which
can result in banana-like shapes.

------------------------------------------
Concatenation of Uncertain Transformations
------------------------------------------

There are two different ways of defining uncertainty of transformations when
we concatenate multiple transformations. We can define uncertainty
in the global frame of reference or in the local frame of reference
and it makes a difference.

Global frame of reference
(:func:`~pytransform3d.uncertainty.concat_globally_uncertain_transforms`):

.. math::

   Exp(_C\boldsymbol{\xi'}) \overline{\boldsymbol{T}}_{CA} = Exp(_C\boldsymbol{\xi}) \overline{\boldsymbol{T}}_{CB} Exp(_B\boldsymbol{\xi}) \overline{\boldsymbol{T}}_{BA},

where :math:`_B\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_{BA})`,
:math:`_C\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_{CB})`,
and :math:`_C\boldsymbol{\xi'} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_{CA})`.

This method of concatenating uncertain transformation is used in Example
:ref:`sphx_glr__auto_examples_plots_plot_concatenate_uncertain_transforms.py`,
which illustrates how the banana distribution is generated.

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_concatenate_uncertain_transforms_001.png
   :target: ../_auto_examples/plots/plot_concatenate_uncertain_transforms.html
   :align: center

Local frame of reference
(:func:`~pytransform3d.uncertainty.concat_locally_uncertain_transforms`):

.. math::

   \overline{\boldsymbol{T}}_{CA} Exp(_A\boldsymbol{\xi'}) = \overline{\boldsymbol{T}}_{CB} Exp(_B\boldsymbol{\xi}) \overline{\boldsymbol{T}}_{BA} Exp(_A\boldsymbol{\xi}),

where :math:`_B\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_B)`,
:math:`_A\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_A)`,
and :math:`_A\boldsymbol{\xi'} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_{A,total})`.

This method of concatenating uncertain transformations is used in Example
:ref:`sphx_glr__auto_examples_visualizations_vis_probabilistic_robot_kinematics.py`,
which illustrates probabilistic robot kinematics.

.. figure:: ../_auto_examples/visualizations/images/sphx_glr_vis_probabilistic_robot_kinematics_001.png
   :target: ../_auto_examples/visualizations/vis_probabilistic_robot_kinematics.html
   :align: center

-------------------------
Fusion of Uncertain Poses
-------------------------

Fusing of multiple uncertain poses with
:func:`~pytransform3d.uncertainty.pose_fusion` is required, for instance,
in state estimation and sensor fusion.
The Example :ref:`sphx_glr__auto_examples_plots_plot_pose_fusion.py`
illustrates this process.

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_pose_fusion_001.png
   :target: ../_auto_examples/plots/plot_pose_fusion.html
   :width: 50%
   :align: center
