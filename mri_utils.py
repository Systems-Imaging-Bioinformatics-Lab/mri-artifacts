import numpy as np
import matplotlib.pyplot as plt


def plot_img_arr(imgArr,cutPt = None,nameList = None,cMapName = 'gray',outName=None,ctrArr =None,ctrNames=None,
                axName = ['Coronal','Sagittal','Axial'],dpi = 72,alpha=1,ctrCols=None):
    permMat = [[2,1,0],[2,0,1],[0,1,2]]
    if len(imgArr.shape) == 3:
        dim = imgArr.shape
        imgArr = np.reshape(imgArr,(dim[0],dim[1],dim[2],1))
    nM = imgArr.shape[3]
    f, axs = plt.subplots(nM,3,figsize=(15,(nM)*5),dpi = dpi)
    axs = np.reshape(axs,(-1,3))
    if cutPt is None:
        dim =  np.array(imgArr.shape[0:3])
        cutPt = (dim/2).astype(int)
    
    for dNo in range(3):
        for mNo in range(imgArr.shape[3]):
            cMat = imgArr[:,:,:,mNo]
            pMat = np.transpose(cMat,permMat[dNo])
            h = axs[mNo,dNo].imshow(pMat[:,:,cutPt[dNo]],origin='lower', cmap=plt.get_cmap(cMapName))
            
            if ctrArr is not None: # get ready to plot the contours if provided
                import cv2 as cv
                if len(ctrArr.shape) == 3:
                    ctrArr = np.reshape(ctrArr,list(ctrArr.shape) + [1])
                if ctrNames is None:
                    ctrNames = ['%i' % idx for idx in range(ctrArr.shape[3])]
                ctrDict = {}
                for cNo in range(ctrArr.shape[3]):
                    ctrPMat = np.transpose(ctrArr[:,:,:,cNo],permMat[dNo])
                    ctrSlice = ctrPMat[:,:,cutPt[dNo]]
                    spacer = np.empty((1,1,2))
                    spacer[:] = np.NaN
                    emptyList = np.count_nonzero(ctrSlice) > 0
                    if np.count_nonzero(ctrSlice) > 0:
                        ctrPts, _ = cv.findContours(ctrSlice.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        temp = [np.concatenate((ctr,ctr[:1,:,:],spacer),axis=0) for ctr in ctrPts]
                        if temp:
                            ctrPts = np.concatenate(temp,axis=0)
                            ctrDict[ctrNames[cNo]] = ctrPts
                            emptyList = False
                        else:
                            emptyList = True
                    if emptyList == True: 
                        cStr = '* %s' % (ctrNames[cNo])
                        ctrDict[cStr] = spacer
                if ctrCols is None:
                    for lblN,ctr in ctrDict.items():
                        axs[mNo,dNo].plot(ctr[:,:,0],ctr[:,:,1],label=lblN,alpha=alpha)
                else:
                    for (lblN,ctr,col) in zip(ctrDict.keys(),ctrDict.values(),ctrCols):
                        axs[mNo,dNo].plot(ctr[:,:,0],ctr[:,:,1],label=lblN,alpha=alpha,color = col)
                if dNo in (0,1,2):
                    axs[mNo,dNo].legend()                    
            if nameList is not None:
                axs[mNo,dNo].set_title('%s %s' % (axName[dNo], nameList[mNo]))
            else:
                axs[mNo,dNo].set_title('%s %i' % (axName[dNo], mNo))
                
#             if dNo == 2:
#                 axs[mNo,dNo].invert_yaxis()
            f.colorbar(h, ax=axs[mNo,dNo])
    if outName is not None:
        f.savefig(outName,dpi=dpi)
        plt.close(f)
        return
    return f, axs

def get_min_background(imgArr):
    dim = list(imgArr.shape)
    if len(imgArr.shape) == 3:
        dim = dim + [1]
        imgArr = np.reshape(imgArr,(dim[0],dim[1],dim[2],1))
        
    for mNo in range(dim[3]):
        mVal = np.amin(imgArr[:,:,:,mNo])
        adjVal = np.std(imgArr[:,:,:,mNo]) / 5
        if mNo == 0:
            bgMask = imgArr[:,:,:,mNo]<=(mVal + adjVal)
        else:
            bgMask = np.logical_and(bgMask,imgArr[:,:,:,mNo]<=(mVal + adj_val))
    dim[3] = 1
    bgMask = np.reshape(bgMask,dim[0:3])
#     print(dim)
    return(bgMask)
        