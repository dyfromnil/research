from PIL import Image
from scipy.io import loadmat
import glob
import os

path=sorted(glob.glob("/root/experiments/My/interpolartion-cycGan/Cardiac/pat/*.mat"))


num_dict={}

count=1
for pa in path:
    print(count)
    count+=1
    m=loadmat(pa)
    pa=pa.split('pat')
    pa=pa[len(pa)-1].split('.')
    num=pa[0]

    if count<=28:
        for t in range(m['sol_yxzt'].shape[3]):
            for z in range(m['sol_yxzt'].shape[2]):
                img=Image.fromarray(m['sol_yxzt'][:,:,z,t]).convert('L')
                if (z+2) < m['sol_yxzt'].shape[2]:
                    img.save("/root/experiments/My/interpolartion-cycGan/data/train/"+num+"_"+str(t)+"_"+str(z)+".png",'png')
                else:
                    img.save("/root/experiments/My/interpolartion-cycGan/data/residual/"+num+"_"+str(t)+"_"+str(z)+".png",'png')
                num_dict[num]=m['sol_yxzt'].shape[2]
    else:
        for t in range(m['sol_yxzt'].shape[3]):
            for z in range(m['sol_yxzt'].shape[2]):
                img=Image.fromarray(m['sol_yxzt'][:,:,z,t]).convert('L')
                if (z+2) < m['sol_yxzt'].shape[2]:
                    img.save("/root/experiments/My/interpolartion-cycGan/data/test/"+num+"_"+str(t)+"_"+str(z)+".png",'png')
                else:
                    img.save("/root/experiments/My/interpolartion-cycGan/data/residual/"+num+"_"+str(t)+"_"+str(z)+".png",'png')
                num_dict[num]=m['sol_yxzt'].shape[2]
