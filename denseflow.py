import os
import sys
import argparse
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image
from skvideo import io
from scipy import misc


def ToImg(raw_flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255, mean=0)
    :param bound: lower and upper bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= -bound
    flow *= (255/float(2*bound))
    return flow


def save_flows(flows, image, save_dir, num, bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)

    if not os.path.exists(os.path.join(out_folder, save_dir)):
        os.makedirs(os.path.join(out_folder, save_dir))

    # save the image
    save_img = os.path.join(out_folder, save_dir,
                            'img_{:05d}.jpg'.format(num))
    image = Image.fromarray(image)
    image = image.resize((340, 256))
    misc.imsave(save_img, image)

    # save the flows
    save_x = os.path.join(out_folder, save_dir,
                          'flow_x_{:05d}.jpg'.format(num))
    save_y = os.path.join(out_folder, save_dir,
                          'flow_y_{:05d}.jpg'.format(num))
    flow_x_img = Image.fromarray(flow_x)
    flow_x_img = flow_x_img.resize((340, 256))
    flow_y_img = Image.fromarray(flow_y)
    flow_y_img = flow_y_img.resize((340, 256))
    misc.imsave(save_x, flow_x_img)
    misc.imsave(save_y, flow_y_img)
    return 0


def dense_flow(*args):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name, save_dir, step, bound = args
    video_file = os.path.join(src_folder, video_name)

    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support
    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    try:
        videocapture = io.vread(video_file)
    except:
        print '{} read error! '.format(video_name)
        return 0

    if videocapture.sum() == 0:
        # if extract nothing, exit!
        print 'Could not initialize capturing', video_name
        exit()

    len_frame = len(videocapture)
    frame_num = 0
    image, prev_image, gray, prev_gray = None, None, None, None
    num0 = 0

    while True:
        if num0 >= len_frame:
            break

        frame = videocapture[num0]
        num0 += 1

        if frame_num == 0:
            image = np.zeros_like(frame)
            gray = np.zeros_like(frame)
            prev_gray = np.zeros_like(frame)
            prev_image = frame
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
            frame_num += 1
            # to pass the out of stepped frames
            step_t = step
            while step_t > 1:
                num0 += 1
                step_t -= 1
            continue

        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frame_0 = prev_gray
        frame_1 = gray
        # default choose the tvl1 algorithm
        dtvl1 = cv2.createOptFlow_DualTVL1()
        flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
        # this is to save flows and img.
        save_flows(flowDTVL1, image, save_dir, frame_num, bound)
        prev_gray = gray
        prev_image = image
        frame_num += 1
        # to pass the out of stepped frames
        step_t = step
        while step_t > 1:
            num0 += 1
            step_t -= 1


def get_video_list(src_folder):
    video_list = [v for v in os.listdir(src_folder) if v != '.DS_Store']
    return video_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="densely extract the video frames and optical flows")
    parser.add_argument('--src_folder', type=str)
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--num_workers', default=1, type=int,
                        help='num of workers to act multi-process')
    parser.add_argument('--step', default=1, type=int, help='gap frames')
    parser.add_argument('--bound', default=15, type=int,
                        help='set the maximum of optical flow')
    parser.add_argument('--s_', default=0, type=int, help='start id')
    parser.add_argument('--e_', default=None, type=int, help='end id')
    parser.add_argument('--debug', default=False, action="store_true",
                        help='use this param to debug')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    src_folder = args.src_folder
    out_folder = args.out_folder

    video_list = get_video_list(src_folder)
    print("find {} videos".format(len(video_list)))

    s_ = args.s_
    if not args.e_:
        e_ = len(video_list)
    video_list = video_list[s_: e_]
    print("ready to process {} videos".format(len(video_list)))
    img_dirs = [video.split('.')[0]
                for video in video_list if video != '.DS_Store']

    if args.debug:
        """
        debug mode
        """
        dense_flow(video_list[0], img_dirs[0], args.step, args.bound)
    else:
        """
        multi run
        """
        pass
