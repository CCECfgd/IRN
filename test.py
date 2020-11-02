import torch
import torchvision
import torch.optim
import networks
import numpy as np
from PIL import Image
import glob
import time
import os


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = data_hazy.resize((640, 480))
    data_hazy = data_hazy.convert("RGB")
    # data_hazy = data_hazy.resize((width, height),Image.ANTIALIAS)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)
    i = 6
    dehaze_net = networks.IRDN(i).cuda()
    s = ""
    dehaze_net.load_state_dict(torch.load('trained_model/i6-outdoor-MSE+SSIM/IRDN.pth'))


    clean_image, _ = dehaze_net(data_hazy)
    # temp_tensor = clean_image[0].cuda().data.cpu().numpy()
    temp_tensor = (clean_image, 0)
    # clean_image = HDR.test(temp_tensor)

    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results_mish/" + image_path.split("/")[-1])
    # torchvision.utils.save_image((data_hazy,0), "results_mish/" + image_path.split("/")[-1])

    # clean_image = Image.fromarray(clean_image)
    temp = image_path.split("/")[-1].replace(r"test dataset/indoor sym haze", "")
    # 	# clean_image.save("results_mish增强/"+temp.replace("\\",""))
    # cv2.imwrite("results_mish/" + temp.replace("合成","",temp),clean_image)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results_mish/" + image_path.split("/")[-1])
    if not os.path.exists('test_result/indoor/' + s):
        os.makedirs('test_result/indoor/' + s)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "对比/i3-mish/" + temp.replace("\\",""))
    torchvision.utils.save_image(clean_image, 'test_result/compare/MSE+SSIM' + s + '/' + temp)

if __name__ == '__main__':

    test_list = glob.glob(r"dataset/test_data/duibi/real/*")
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    for image in test_list:
        dehaze_image(image)

        print(image, "done!")
