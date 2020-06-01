# visualization image

from scipy import misc 
import scipy 

def visul(img, j, name):
    img = img.data.cpu().numpy()
    for i in range(16):
        savepath = '/data/zjj/focus/code/resnet_torch/vis_imgs/'+name+'/'+str(i)+'.jpg'
        scipy.misc.imsave(savepath, img[i,:,:,:].transpose(1,2,0))




import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
def hist(cam):
    cam = cam.data.cpu().numpy()
    d = cam.reshape(-1)
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('the Histogram of cam value')
    plt.savefig('/data/zjj/focus/code/resnet_torch/vis_imgs/camhist.png')