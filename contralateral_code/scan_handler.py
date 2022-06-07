import os
import numpy as np
import nibabel as nib
from scipy.stats import iqr
from copy import deepcopy

def default_modal_list():
    return ['FLAIR','T1', 'T1post','T2']
def default_modal_dict():
    return {'t1': 't1.nii.gz', 't1post':'t1Gd.nii.gz', 't2':'t2.nii.gz', 'flair': 'flair.nii.gz'}

def norm_modals(cDir,modal_list = ['FLAIR','T1', 'T1post','T2'], modal_dict = {'t1':'t1.nii.gz', 't1post':'t1Gd.nii.gz', 't2':'t2.nii.gz', 'flair': 'flair.nii.gz'},label_fname = 'truth.nii.gz',label_img = None):
    img = {modal:[] for modal in modal_list}
   

    if label_img is None:
        cLblFile = os.path.join(cDir,label_fname)
        if not os.path.exists(cLblFile):
            return img
        label_img = nib.load(cLblFile)
        
    label_arr = label_img.get_fdata().astype(np.float64)
    
    for modal in modal_list:
        ml = modal.lower()
        cNIIFile = os.path.join(cDir,modal_dict[ml])
        img = {}
        inten_dic = {}

        for modal in modal_list:
            #load image
            if not os.path.exists(cNIIFile):
                img[modal] = []
                continue
            cImg = nib.load(cNIIFile)
            # put array into single precision, or else the downstream pyradiomics freaks out
            cArr = np.float32(cImg.get_data())
            
            # assumption that the scans have been skull-stripped
            brain_mask = cArr > 0
#             brain_mask_nt = np.logical_and(brain_mask, np.logical_not(np.logical_or(label_arr,contr_label )))
            brain_mask_nt = np.logical_and(brain_mask, label_arr)
            # median IQR normalization
            median = np.median(cArr[brain_mask_nt])
            curr_iqr = iqr(cArr[brain_mask_nt])
            cArrNorm = deepcopy(cArr)
            cArrNorm =  (cArrNorm-median)/curr_iqr

            img[modal] = cArrNorm
    return img