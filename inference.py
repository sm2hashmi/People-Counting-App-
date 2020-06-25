#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):
        ### TODO: Load the model ###
        load_model_xml = model
        load_model_bin = os.path.splitext(load_model_xml)[0] + ".bin"
        
        while not plugin:
            log.info("Please wait. Starting plugin for {} device... ".format(device))
            self.plugin = IECore()
        
        else:
            self.plugin = plugin
        
        if cpu_extension and CPU in device:
            self.plugin.add_cpu_extension(cpu_extension)
        
        log.info('Reading IR, Please wait.')
        self.net = IENetwork(model = load_model_xml, weights = load_model_bin)
        log.info('Completed. Loading IR to the plugin. This may take some time')
        ### TODO: Check for supported layers ###
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
            unsupported_layers = [l for l in self.net.layers.key() if l not in supported_layers]
        
            if len(unsupported_layers) != 0:
                log.error('There are a number of unsupported layers found: '.format(unsupported_layers))
                sys.exit(1)
        
        if num_request == 0:
            self.net_plugin = self.plugin.load(network = self.net)
            
        else:
            self.net_plugin = self.plugin.load(network = self.net, num_requests = num_requests)
        
        self.input_blob = next(iter(self.net.input))
        self.output_blob = next(iter(self.net.output))
        
        if len(self.net.inputs.key()) == input_size:
            log.error('Sorry, this app supports {} input topologies. Please make the necessary changes and try again'.format(len(self.net.inputs)))
            sys.exit(1)
        
        if len(self.net.outputs) == output_size:
            log.error('Sorry, this app supports {} output topologies. Please make the necessary changes and try again'.format(len(self.net.inputs)))
            sys.exit(1)
        
        return self.plugin, self.get_input_shape
            
            
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input.blob].shape

    def exec_net(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request = self.net_plugin.start_async(request_id = request_id, inputs = {self.input_blob: frame})
       
        return self.net_plugin

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        
        wait_status = self.net_plugin.requests[request_id].wait(-1)
        
        return wait_status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        
        if output:
            result = self.infer_request.outputs[output]
        
        else:
            result = self.net_plugin.requests[request_id].outputs[self.output_blob]
        
        return result
    
    def delete_instances(self):
        del self.net_plugin
        del self_plugin
        del self.net

