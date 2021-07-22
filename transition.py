

import os
import torch
from torchvision import transforms
from inference.Inferencer import Inferencer
from models.PasticheModel import PasticheModel
from PIL import Image
from glob import glob
import numpy as np
import cv2

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def export_as_video(inference, path_img, content1, content2, path_video, total_frames, CLIP_FPS):

    #imgname = "sample.jpg"
    #content1, content2 = 0, 3
    #filepath = 'test.mp4'
    #CLIP_FPS = 30.0
    #total_frames = 100

    im = Image.open(path_img).convert('RGB')
    w, h = im.size
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path_video, codec, CLIP_FPS, (w, h))

    for loop in range(total_frames):

        #PIL image
        #First Style (0-15)
        #Second Style (0-15)
        #Percentage mixture between the two styles (0.0-1.0)
        img_trans = inference.eval_image(im, content1, content2, loop / float(total_frames))

        video.write(pil2cv(img_trans))

    video.release()


def main(args):

    styles_dir = args.styles_dir
    model_dir=args.model_dir
    image_size = args.imsize
    path_image = args.path_image
    content1 = args.content1
    content2 = args.content2
    video_name = args.path_video
    total_frames = args.total_frames
    CLIP_FPS = 30.0
    loop = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #num_styles = 16
    #
    
    style_images_dir = glob(os.path.join(styles_dir, '*.jpg'))
    num_styles = len(style_images_dir)

    model_save_dir = model_dir + "/pastichemodel_"+str(loop)+"-FINAL.pth"
    pastichemodel = PasticheModel(num_styles)
    inference = Inferencer(pastichemodel, device, image_size)
    inference.load_model_weights(model_save_dir)


    export_as_video(inference, path_image, content1, content2, video_name, total_frames, CLIP_FPS)




if __name__ == '__main__':

    import argparse

    main_arg_parser = argparse.ArgumentParser(description="parser for training mutli-style-transfer")
    
    main_arg_parser.add_argument("--styles-dir", type=str, required=True,
                                  help="path to folder containing style images")
    
    main_arg_parser.add_argument("--model-dir", type=str, default=None,
                                  help="directory to save the model in")

    main_arg_parser.add_argument("--imsize", type=int, default=512,
                                  help="")

    main_arg_parser.add_argument("--content1", type=int, default=0,
                                  help="")
    main_arg_parser.add_argument("--content2", type=int, default=1,
                                  help="")

    main_arg_parser.add_argument("--path_video", type=str, default='result/test.mp4',
                                  help="")
    main_arg_parser.add_argument("--path_image", type=str, default='dataset/images/trump.jpg',
                                  help="")
    main_arg_parser.add_argument("--total_frames", type=int, default=1,
                                  help="")

    args = main_arg_parser.parse_args()

    main(args)
    


    