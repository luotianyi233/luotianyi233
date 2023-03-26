import torch
import tensorboardX as tdx
import tqdm
import argparse
from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='edit tensorboard file')
    parser.add_argument('--input', type=str, help='The path of input tensorboard file')
    parser.add_argument('--output', type=str, default="", help='The dir of output tensorboard file')
    parser.add_argument('--tag_old', type=str, default="", help='The old tag which want to change name')
    parser.add_argument('--tag_new', type=str, help='The old tag which want to change name')
    args = parser.parse_args()
    change_tag = False
    if args.output == "":
        output = args.input+"_changed"
    else:
        output = args.output
    writer = tdx.SummaryWriter(output)
    print("Input: {}\nOutput: {}".format(args.input, output))
    if args.tag_old != '':
        print("old tag \"{}\" -> new tag \"{}\" ".format(args.tag_old, args.tag_new))
    # writer = tf.summary.SummaryWriter(args.output)

    total=None
    for event in tqdm.tqdm(summary_iterator(args.input), total=total):
        for value in event.summary.value:
            tag = value.tag
            if tag.find("RESULT") != -1 or tag.find("IMAGE") != -1:    # ignore image
                continue
            # if tag.find("Epoch") != -1:
            #     print(tag)
            if tag == args.tag_old:         # change tag name
                change_tag=True
                tag = args.tag_new
            writer.add_scalar(tag, value.simple_value, event.step, walltime=event.wall_time)
            # print("{}\t{}\t{}".format(value.tag, value.simple_value, event.step))
    writer.close()

    if change_tag:
        print("Tag changed")
