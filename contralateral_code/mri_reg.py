import nibabel as nib
import numpy as np
from nilearn import datasets
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration,transform_centers_of_mass)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)

def vox_coords(aff_in, i, j, k):
    M = aff_in[:3, :3]
    abc = aff_in[:3, 3]
    """ Return X, Y, Z coordinates for i, j, k """
    return M.dot([i, j, k]) + abc

def ms_affreg(moving_img,template_img=None,level_iters = [250, 150, 75],sigmas = [3.0, 1.0, 0.0],factors = [4, 2, 1],
             nbins = 32,sampling_prop = None,reg_type = 3):
    """ FROM: https://dipy.org/documentation/1.0.0./examples_built/affine_registration_3d/
    and FROM: https://bic-berkeley.github.io/psych-214-fall-2016/dipy_registration.html
    """
    
    metric = MutualInformationMetric(nbins, sampling_prop)
    if template_img == None:
        ### https://nilearn.github.io/modules/generated/nilearn.datasets.load_mni152_template.html
        ### more datasets https://nilearn.github.io/auto_examples/01_plotting/plot_prob_atlas.html
        mni = datasets.load_mni152_template()
        template_img = mni
    template_data = template_img.get_data()
    template_affine = template_img.affine
    
    moving_data = moving_img.get_data()
    moving_affine = moving_img.affine
    """
    We can obtain a very rough (and fast) registration by just aligning the centers
    of mass of the two images
    """
    c_of_mass = transform_centers_of_mass(template_data, template_affine,
                                      moving_data, moving_affine)

    
    
    affreg = AffineRegistration(metric=metric,level_iters=level_iters,
                                sigmas=sigmas,factors=factors)
    """
    Using AffineRegistration we can register our images in as many stages as we
    want, providing previous results as initialization for the next (the same logic
    as in ANTS). The reason why it is useful is that registration is a non-convex
    optimization problem (it may have more than one local optima), which means that
    it is very important to initialize as close to the solution as possible. For
    example, lets start with our (previously computed) rough transformation
    aligning the centers of mass of our images, and then refine it in three stages.
    First look for an optimal translation. The dictionary regtransforms contains
    all available transforms, we obtain one of them by providing its name and the
    dimension (either 2 or 3) of the image we are working with (since we are
    aligning volumes, the dimension is 3)
    """
    print('Translation Registration (1/%i)' % reg_type)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine,
                                    starting_affine=starting_affine)
    if reg_type == 1:
        return (translation,template_img)
    """
    Now lets refine with a rigid transform (this may even modify our previously
    found optimal translation)
    """
    print('Rigid Registration (2/%i)' % reg_type)
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine,
                                    starting_affine=starting_affine)
    if reg_type == 2:
        return (rigid,template_img)
    """
    Finally, lets refine with a full affine transform (translation, rotation, scale
    and shear), it is safer to fit more degrees of freedom now, since we must be
    very close to the optimal transform
    """
    print('Affine Registration (3/%i)' % reg_type)
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                             template_affine, moving_affine,
                             starting_affine=starting_affine)
    return (affine,template_img)

def resize_template(template_img,moving_img):
    mov_scale = np.multiply(np.abs(np.diag(moving_img.affine)[0:3]),[-1,1,1]) # orientation of the MNI template but the scale of the scan
    size_diff = np.divide(template_img.shape,moving_img.shape)
    scale_diff = np.divide(np.diag(template_img.affine)[0:3],np.diag(moving_img.affine)[0:3])
    sz_sc_diff = np.abs(np.multiply(size_diff,scale_diff))

    cent_idxs = vox_coords(np.linalg.inv(template_img.affine),0,0,0)
    cent_perc = np.divide(np.array(cent_idxs),template_img.shape)
    # cent_rs_vals = np.multiply(-np.multiply(cent_perc,moving_img.shape),mov_scale)
    cent_rs_vals = np.multiply(-np.divide(cent_idxs,size_diff),mov_scale)

    # print(cent_idxs,cent_rs_vals)

    M = np.diag(mov_scale)
    output_aff = nib.affines.from_matvec(M ,cent_rs_vals)
    # print(output_aff,template_img.affine)
    rs_affine = AffineMap(nib.affines.from_matvec(np.diag(sz_sc_diff),[0,0,0]),
                          moving_img.shape, output_aff,
                          template_img.shape, template_img.affine)

#     print(rs_affine)
    rs_temp = rs_affine.transform(template_img.get_data())
#     print(rs_temp.shape)

    rs_temp_nib = nib.nifti1.Nifti1Image(rs_temp,output_aff)
    
#     c_of_mass = transform_centers_of_mass(rs_temp, output_aff,
#                                       template_img.get_data(), template_img.affine)
#     print(c_of_mass)

#     f, ax = plt.subplots(figsize=(9, 6))
#     pl_anat = plotting.plot_anat(template_img,axes=ax,cut_coords=(0,0,0))
#     plt.show()
#     f, ax = plt.subplots(figsize=(9, 6))
#     pl_anat = plotting.plot_anat(rs_temp_nib,axes=ax,cut_coords=(0,0,0))
#     com_temp = c_of_mass.transform(template_img.get_data())
#     plt.show()
#     regtools.overlay_slices(com_temp, rs_temp, None, 0,
#                             "Static", "Moving", "resampled_0.png")
#     regtools.overlay_slices(com_temp, rs_temp, None, 1,
#                             "Static", "Moving", "resampled_1.png")
#     regtools.overlay_slices(com_temp, rs_temp, None, 2,
#                             "Static", "Moving", "resampled_2.png")
    
    return rs_temp_nib