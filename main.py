"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

#from pydub import AudioSegment
#from pydub.playback import play
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_rectangular_box(frame, result, initial_width, initial_height, prob_threshold):
    
    present_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            starting_point = int((obj[3]*initial_width), (obj[4]*initial_height))
            ending_point = int((obj[5]*initial_width), (obj[6]*initial_height))
            box_colour = (255, 10, 0)
            box_thickness = 3
            cv2.rectangle(frame, starting_point, ending_point, box_colour, box_thickness)
            present_count += 1
        
        return frame, present_count

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    #Initial, global variables for counting
    current_request_id = 0
    start_time = 0
    last_count = 0
    total_count = 0
    
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ###  Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, current_request_id, args.cpu_extension)
    model_input_shape = infer_network.get_input_shape()

    ### Handle the input stream ###
    single_image_mode = False
    
    while args.input == 'CAM':
        input_stream = 0
        
    if args.input.endswith('.jpg') or args.input.endswith('.png') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
        
    else:
        input_stream = args.input
        assert os.path.isfile(args.input),"The input file does not exist"
        
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(input_stream)
        
    if not cap.IsOpened():
        log.error('Error! The video file/source is not opening' )
    
    #inital width and height taken from the input
    initial_width = int(cap.get(3))
    initial_height = int(cap.get(4))
     ###  Loop until stream is over ###   
    while cap.isOpened():
         ###  Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
            
        pressed_key = cv2.waitKey(60)
        ### Pre-process the image as needed ###
        width = model_input_shape[3]
        height = model_input_shape[2]
        processed_input_image = cv2.resize(frame,(width, height))
        processed_input_image = processed_input_image.transpose((2, 0, 1))
        processed_input_image = processed_input_image.reshape(model_input_shape[0], model_input_shape[1], height, width)
        ###  Start asynchronous inference for specified request ###
        start_of_inference = time.time()
        infer_network.exec_net(current_request_id, processed_input_image)
        
        ###  Wait for the result ###
        if infer_network.wait(current_request_id) == 0:
            detection_time = int(time.time() - start_of_inference) * 1000
            ###  Get the results of the inference request ###
            result = infer_network.get_output(current_request_id)
            ### Extract any desired stats from the results ###
            frame, present_count = draw_rectangular_box(frame, result, initial_width, initial_height, prob_threshold)
            ##Find out the inference time and write the result on the video as text.
            inf_time_msg = "Inference time: {:.5f}ms".format(detection_time)
            cv2.putText(frame, inf_time_msg, (20,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            #Person's count is calculated here
            if present_count > last_count:
                start_time = time.time()
                total_count += present_count - last_count
                client.publish('person', json.dumps({"total": total_count}))
            #Duration is calculated here
            if present_count < last_count:
                person_duration = int(time.time() - start_time)
                # This is to prevent double counting. Higher value to ensure that the app does not get oversensitive#
                if person_duration > 5:
                    total_count -= 1
                client.publish('person/duration', json.dumps({"duration": person_duration}))
            
                #if present_count >=4:
                #print('Alert! Number of people exceeds the limit! Please take necessary action.')
                
                
            client.publish('person', json.dumps({"count": present_count}))
            last_count = present_count
            # End if escape key is pressed
            if pressed_key == 27:
                break
         ###  Send the frame to the FFMPEG server ###    
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imWrite('output_image.jpg', frame)
        
        cap.release()
        cv2.DestroyAllWindows()
        client.disconnect()
        infer_network.clean()
        
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
