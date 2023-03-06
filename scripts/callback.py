import math
import os
import sys
import traceback
import requests
import io
import base64
import urllib3
import threading
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images


class Script(scripts.Script):
    t = 0

    def title(self):
        return "callback"

    def ui(self, is_img2img):
        callback_url = gr.Textbox(label="callback_url", lines=1)

        return [callback_url]

    def send(self, image, callback_url, gen_time):
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format="PNG")
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()

        base64_encoded_result_bytes = base64.b64encode(img_bytes)

        if callback_url:
            requests.post(callback_url, data={
                'base64_image': base64_encoded_result_bytes.decode('ascii'),
                'gen_time': gen_time,
                'end_time': time.time()
            }, verify=False)

    def process_batch(self, p=None, batch_number=None, prompts=None, seeds=None, subseeds=None):
        self.t = time.time()

    def postprocess_batch(self, p=None, x_samples_ddim=None, batch_number=None):
        self.t = time.time() - self.t

    def run(self, p, callback_url):
        p.scripts.process_batch = self.process_batch
        p.scripts.postprocess_batch = self.postprocess_batch

        proc = process_images(p)
        image = proc.images

        t = threading.Thread(target=self.send, args=(image[0], callback_url, self.t))
        t.daemon = True
        t.start()

        return Processed(p, image, p.seed, proc.info)