# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str,
                    help="The path of the input image.", default="./data/demo_data/messi.jpg")
# parser.add_argument("input_image", type=str,
#                     help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3_revised.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)


# Suppose executing it on cloud
def model_splitter(sess, ckpt_dir, model_scope=['inference_1', 'inference_2'], save_dir='./partial_model'):
    '''
    :param sess: tf.Session()
    :param ckpt_dir: str, Directory of ckpt file of entire model weights
    :param model_scope: list, partial model scope
    :param save_dir: str, Director to save partial model
    '''
    if not isinstance(model_scope, (list, tuple)):
        raise TypeError('model_scope is not list or tuple!')
    tmp = tf.get_default_graph().get_collection('variables')
    temp =[]
    for _ in model_scope:
        temp.append([])

    for var in tmp:
        for c, v in enumerate(temp):
            if model_scope[c] in var.op.name:
                v.append(var)

    saver_1 = tf.train.Saver()
    saver_1.restore(sess, ckpt_dir)
    for i in range(len(model_scope)):
        saver_2 = tf.train.Saver(temp[i])
        ckpt_name = save_dir+'_'+str(i).zfill(2)+'.ckpt'
        saver_2.save(sess, ckpt_name)
        print('Splitting the model in {}......Done!'.format(ckpt_name))


def partial_reconstructor(sess,  partial_ckpt_dir, partial_scope='inference_1'):
    '''

    :param sess: tf.Session()
    :param partial_ckpt_dir: dict, Directory of the partial ckpt file
    :param partial_scope: str, scope name of the partial model
    '''
    tmp = tf.get_default_graph().get_collection('variables')
    temp = []
    for var in tmp:
        if partial_scope in var.op.name:
            temp.append(var)
    saver = tf.train.Saver(temp)
    saver.restore(sess, partial_ckpt_dir)
    print('Restoring partial model {}......Done!'.format(partial_scope))


def get_features(sess, feed_dict, feature_name):
    '''
    :param sess: tf.Session()
    :param feed_dict: dict, name of the input ops as a key and input as a value
    :param feature_name: list, a list of names of the output tensors
    :return: result of the output tensor
    '''
    if not isinstance(feature_name, (list, tuple)):
        raise TypeError('model_scope is not list or tuple!')

    temp = []
    for c, v in enumerate(feature_name):
        var = get_tensor(v)
        temp.append(var)

    result = sess.run(temp, feed_dict=feed_dict)
    print('Getting features......Done!')
    return result


def preprocess(img_ori):
    img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    print('Preprocessing.......Done!')
    return img


def get_tensor(ops_name):
    tt = tf.get_default_graph().get_tensor_by_name(ops_name+':0')
    return tt


def main():
    with tf.Session() as sess:
        # (Cloud & Edges) Full model information, construct a graph of the model, need to be called on cloud and on edge
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        _, _, _ = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, iou_thresh=0.5)

        # On Cloud
        # (Cloud) Model splitter
        model_splitter(sess, "./data/darknet_weights/yolov3_revised.ckpt")

        # image reader or video processor
        img_ori = cv2.imread(args.input_image)

        # Preprocess
        height_ori, width_ori = img_ori.shape[:2]
        img = preprocess(img_ori)

        # On Edge 1
        # (Edge1) Load parameters of the partial model 1
        partial_reconstructor(sess, './partial_model_00.ckpt', 'inference_1')

        # (Edge1) Inference 1
        feed_1 = {get_tensor('input_data'): img}  # input tensor as key , input value as value
        out_1 = ['inference_1/feature_output_1', 'inference_1/feature_output_2','inference_1/feature_output_3']  # names of the output tensors
        result_1 = get_features(sess, feed_1, out_1)  # get intermediate features from inference 1

        # Feature map compression simulation
        for i in range(len(result_1)):
            result_1[i][result_1[i]<0] = 0
            result_1[i] = result_1[i]/25*255
            result_1[i] = result_1[i].astype(np.uint8)
        np.savez_compressed('result/messi_inter_feat_quantized',a = result_1[0], b=result_1[1], c=result_1[2])
        for i in range(len(result_1)):
            result_1[i] = result_1[i]/255 *25
            result_1[i] = result_1[i].astype(np.float32)
        # On Edge 2
        # (Edge2) Load parameters of the parital model 2
        partial_reconstructor(sess, './partial_model_01.ckpt', 'inference_2')
        # (Edge2) Inference 2 with post process
        feed_2 = {get_tensor('inference_2/feature_input_1'): result_1[0],
                  get_tensor('inference_2/feature_input_2'): result_1[1],
                  get_tensor('inference_2/feature_input_3'): result_1[2]}
        out_2 = ['result/boxes','result/score','result/label']

        result_2 = get_features(sess, feed_2,out_2)
        boxes_, scores_, labels_ = result_2

        # Show the result image
        boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
        boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
        cv2.imshow('Detection result', img_ori)
        cv2.imwrite('result/messi_quantized.jpg', img_ori)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
