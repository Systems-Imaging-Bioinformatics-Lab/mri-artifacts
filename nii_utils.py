import os
import warnings
import re

import nibabel as nib  # pip install nibabel
import numpy as np

import mri_utils
from scipy.ndimage import zoom


modalList = ['flair','t1w','t1gd','t2w']
reDict = {'label':'GlistrBoost',
         'label_manual': 'GlistrBoost_ManuallyCorrected',
         'flair': 'flair',
         't1w': 't1',
         't1gd': 't1Gd',
         't2w': 't2'}

##  https://stackoverflow.com/a/47269413
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def get_bbox(arr):
    # get the bounding box for where the values != 0
    nD = len(arr.shape)
    mmIdx  = np.empty((nD,2))
    mmIdx[:] = np.NaN
    for axNo in range(nD):
        firstArr = first_nonzero(arr,axis = axNo)
        lastArr = last_nonzero(arr,axis = axNo)
        mmIdx[axNo,0] = np.amin(firstArr[firstArr >= 0])
        mmIdx[axNo,1] = np.amax(lastArr[lastArr >= 0])
    return mmIdx

def crop_resize(img, bbox, cropx, cropy, cropz):
    bbox = bbox.astype(int)
    cropVol = img[bbox[0,0]:bbox[0,1],bbox[1,0]:bbox[1,1],bbox[2,0]:bbox[2,1]]    
    
#     print(cropVol.shape)
    cropDim = np.ones(len(cropVol.shape)) # error handling for 4D inputs
    cropDim[0] = cropx
    cropDim[1] = cropy
    cropDim[2] = cropz
    
    RSMat = np.divide(cropDim,cropVol.shape)
    RSMat = np.tile(np.amin(RSMat),len(RSMat)) # long axis scaling
    cropRS = zoom(cropVol, RSMat,cval = 0)
    cropRS[cropRS<0]= 0 # handle some artifact
    padSz = np.array(cropDim) - np.array(cropRS.shape)
    padSzArr = np.vstack((np.ceil(padSz/2),np.floor(padSz/2))).T.astype(int)
    cropRS = np.pad(cropRS,padSzArr,constant_values = 0)
    return cropRS

def crop_only(img,bbox):
    bbox = bbox.astype(int)
    cropVol = img[bbox[0,0]:bbox[0,1],bbox[1,0]:bbox[1,1],bbox[2,0]:bbox[2,1]]    
    return cropVol


def preprocess_inputs(img,bbox,RSShape = 144,RSTF=True):
    """
    Process the input images

    For BraTS subset:
    INPUT CHANNELS:  "modality": {
         "0": "FLAIR", T2-weighted-Fluid-Attenuated Inversion Recovery MRI
         "1": "T1w",  T1-weighted MRI
         "2": "t1gd", T1-gadolinium contrast MRI
         "3": "T2w"   T2-weighted MRI
     }
    """
    if len(img.shape) != 4:  # Make sure 4D
        img = np.expand_dims(img, -1)
    if RSTF == True:
        img = crop_resize(img,bbox, RSShape,RSShape,RSShape)
    else:
        img = crop_only(img,bbox)
    img[img < 0] = 0 # handle some artifact from zoom
    img = normalize_img(img)

    img = np.swapaxes(np.array(img), 0, -2)

    # img = img[:,:,:,[0]]  # Just get the FLAIR channel

    return img


def preprocess_labels(msk,bbox,RSShape = 144,RSTF=True):
    """
    Process the ground truth labels

    For BraTS subset:
    LABEL_CHANNELS: "labels": {
         "0": "background",  No tumor
         "1": "edema",       Swelling around tumor
         "2": "non-enhancing tumor",  Tumor that isn't enhanced by Gadolinium contrast
         "3": "enhancing tumour"  Gadolinium contrast enhanced regions
     }

    """
    if len(msk.shape) != 4:  # Make sure 4D
        msk = np.expand_dims(msk, -1)

    if RSTF == True:
        msk = crop_resize(msk, bbox, RSShape,RSShape,RSShape)
    else:
        msk = crop_only(msk, bbox)
    msk = msk >.5
    # Combining all masks assumes that a mask value of 0 is the background
#     msk[msk > 1] = 1  # Combine all masks
    msk = np.swapaxes(np.array(msk), 0, -2)

    return msk

def normalize_img(img):
    """
    Normalize the pixel values.
    This is one of the most important preprocessing steps.
    We need to make sure that the pixel values have a mean of 0
    and a standard deviation of 1 to help the model to train
    faster and more accurately.
    """

    for channel in range(img.shape[3]):
        img[:, :, :, channel] = (
            img[:, :, :, channel] - np.mean(img[:, :, :, channel])) \
            / np.std(img[:, :, :, channel])

    return img


def prepare_data(data_path,imOutDir,lblOutDir,qcOutDir=None):
    p_dirs = []
    patients = []
    for root, dirs, files in os.walk(data_path):
        for dn in dirs:
            patients.append(dn)
            p_dirs.append(os.path.join(root,dn))

    for idx in tqdm(range(len(patients))):
        cDir = p_dirs[idx]
        cPt = patients[idx]
        prepare_patient(cDir,cPt,imOutDir,lblOutDir,qcOutDir)
        
def prepare_patient(cDir,cPt,imOutDir=None,lblOutDir=None,qcOutDir=None,modalList=modalList,reDict=reDict,
                    RSTF=True):
    fList = os.listdir(cDir)

    ## find images
    bboxes  = np.empty((3,2,len(modalList)))
    bboxes[:] = np.NaN
    modDim  = np.empty((3,len(modalList)))
    modDim[:] = np.NaN
    imDict = {modal : [] for modal in modalList}
    for mIdx in range(len(modalList)):
        modal= modalList[mIdx]
        matchStr = '(.*)%s\.nii(.gz)?$' % (reDict[modal])

        mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
        if len(mList) == 0:
            warnings.warn('Missing modal: %s, %s' % (modal,cDir))
            return None,None

        if len(mList) > 1:
            warnings.warn('Multiple Matches: %s, %s' % (modal,cDir))
            print(mList)

        data_filename = os.path.join(cDir, mList[0])
        im_nib = nib.load(data_filename)
        img = im_nib.get_fdata()

        imDict[modal] = img
        modDim[:,mIdx] = imDict[modal].shape
        bboxes[:,:,mIdx] = get_bbox(imDict[modal])

    if not np.all(np.equal(np.amax(bboxes,axis = 2),np.amin(bboxes,axis = 2))):
        print(cPt)
        print(bboxes)

    # check to make sure the images are the same size
    if not np.all(np.equal(np.amax(modDim,axis = 1),np.amin(modDim,axis = 1))):
        print(cPt)
        print(modDim)
#     bbox = np.mean(bboxes,axis=2).astype(int)
    bbox = np.vstack((np.amin(bboxes[:,0,:],axis=1),np.amin(bboxes[:,1,:],axis=1))).T
        # find labels
    matchStr = '(.*)%s\.nii(.gz)?$' % (reDict['label_manual'])
    mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
    if len(mList) == 0:
        matchStr = '(.*)%s\.nii(.gz)?$' % (reDict['label'])
        mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
        if len(mList) == 0:
            warnings.warn('Missing label: %s' % (cDir))
            return None, None
    if len(mList) > 1:
        warnings.warn('Multiple Matches: %s' % (cDir))
        print(mList)
    data_filename = os.path.join(cDir, mList[0])
    lbl_nib = nib.load(data_filename)
    lbl = lbl_nib.get_fdata()
    cropLbl = preprocess_labels(lbl,bbox,RSTF=RSTF)

    if lblOutDir is not None: # saving the files
        lblOutName = '%s_label.npy' % (cPt)
        if not os.path.isdir(lblOutDir):
            os.makedirs(lblOutDir)
        np.save(os.path.join(lblOutDir,lblOutName),cropLbl)


    # crop and resize the images
    cropImDict = {modal : [] for modal in modalList}
    for mIdx in range(len(modalList)):
        modal= modalList[mIdx]
        crop_img = preprocess_inputs(imDict[modal],bbox,RSTF=RSTF)
        cropImDict[modal] = crop_img


    cropImg = np.concatenate(list(cropImDict.values()),axis=3)
    if imOutDir is not None: # saving the files
        imOutName = '%s_image.npy' % (cPt)
        if not os.path.isdir(imOutDir):
            os.makedirs(imOutDir)
        np.save(os.path.join(imOutDir,imOutName),cropImg)
    
    if qcOutDir is not None: #making QC files
        imgArr = np.concatenate((cropLbl,cropImg),axis =3)
        nameList = ["Mask"] + modalList
        
        maskCutPt = [np.argmax(np.sum(cropLbl,axis=(1,2))),
                 np.argmax(np.sum(cropLbl,axis=(0,2))),
                 np.argmax(np.sum(cropLbl,axis=(0,1)))]
        outName = os.path.join(qcOutDir,'Mask_QC_%s.png' % cPt)
        if not os.path.isdir(qcOutDir):
            os.makedirs(qcOutDir)
        
        mri_utils.plot_img_arr(imgArr,cutPt=maskCutPt,nameList = nameList,outName=outName)
        
        
        outName = os.path.join(qcOutDir,'Standard_QC_%s.png' % cPt)
        mri_utils.plot_img_arr(imgArr,nameList = nameList,outName=outName)
        
    
    return cropLbl,cropImg


def load_patient(cDir,modalList=modalList,reDict=reDict):
    fList = os.listdir(cDir)

    ## find images
    imDict = {modal : [] for modal in modalList}
    imNibDict = {modal : [] for modal in modalList}
    for mIdx in range(len(modalList)):
        modal= modalList[mIdx]
        matchStr = '(.*)%s\.nii(.gz)?$' % (reDict[modal])

        mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
        if len(mList) == 0:
            warnings.warn('Missing modal: %s, %s' % (modal,cDir))
            return None,None

        if len(mList) > 1:
            warnings.warn('Multiple Matches: %s, %s' % (modal,cDir))
            print(mList)

        data_filename = os.path.join(cDir, mList[0])
        im_nib = nib.load(data_filename)
        img = im_nib.get_fdata()

        imDict[modal] = img
        imNibDict[modal] = im_nib
        
    matchStr = '(.*)%s\.nii(.gz)?$' % (reDict['label_manual'])
    mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
    if len(mList) == 0:
        matchStr = '(.*)%s\.nii(.gz)?$' % (reDict['label'])
        mList = [f for f in fList if re.match(matchStr,f,flags=re.IGNORECASE)]
        if len(mList) == 0:
            warnings.warn('Missing label: %s' % (cDir))
            return None, None
    if len(mList) > 1:
        warnings.warn('Multiple Matches: %s' % (cDir))
        print(mList)
    data_filename = os.path.join(cDir, mList[0])
    lbl_nib = nib.load(data_filename)
    lbl = lbl_nib.get_fdata()
    
    return lbl_nib, imNibDict