import numpy as np, cv2, PIL.Image as I
m=(np.array(I.open('nmask.png'))>0).astype(np.uint8)
# vertical run length per pixel
vr=np.zeros_like(m,dtype=np.int32)
H,W=m.shape
for x in range(W):
    col=m[:,x]; y=0
    while y<H:
        if col[y]:
            y2=y
            while y2+1<H and col[y2+1]: y2+=1
            vr[y:y2+1,x]=y2-y+1
            y=y2+1
        else: y+=1
line=np.zeros_like(m)
for y in range(H):
    row=m[y]; x=0
    while x<W:
        if row[x]:
            x2=x
            while x2+1<W and row[x2+1]: x2+=1
            L=x2-x+1
            if L>=22:
                seg=slice(x,x2+1)
                thin=vr[y,seg]<=8
                line[y,seg]=thin.astype(np.uint8)
            x=x2+1
        else: x+=1
ink=(m>0)&(line==0)
ink=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
I.fromarray((ink*255).astype(np.uint8)).save('nink2.png')
print('line px',line.sum(),'ink',ink.sum())
