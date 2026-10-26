import numpy as np, PIL.Image as I, cv2
img=np.load('nimg.npy').astype(int)
lum=img.mean(2)
m=(lum<170).astype(np.uint8)
dt=cv2.distanceTransform(m,cv2.DIST_L2,5)
thick=2*dt   # approx stroke thickness
H,W=m.shape
remove=np.zeros_like(m)
for y in range(H):
    row=m[y]; x=0
    while x<W:
        if row[x]:
            x2=x
            while x2+1<W and row[x2+1]: x2+=1
            L=x2-x+1
            if L>=20:
                t=np.median(thick[y,x:x2+1])
                if t<=6.5:
                    remove[y,x:x2+1]=1
            x=x2+1
        else: x+=1
ink=(m>0)&(remove==0)
ink=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
I.fromarray((ink*255).astype(np.uint8)).save('ink3.png')
print('removed',remove.sum(),'ink',ink.sum())
