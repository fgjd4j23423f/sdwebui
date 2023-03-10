import requests
import io
import base64
import urllib3
import threading
import time
import json
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image, ImageEnhance

class Script(scripts.Script):
    t = 0

    def title(self):
        return "callback"

    def ui(self, is_img2img):
        callback_url = gr.Textbox(label="callback_url", lines=1)
        filters = gr.Textbox(label="filters", lines=1)

        return [callback_url, filters]

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

    def run(self, p, callback_url, filters):
        p.scripts.process_batch = self.process_batch
        p.scripts.postprocess_batch = self.postprocess_batch

        proc = process_images(p)
        image = proc.images

        filters = json.loads(filters)
        for filter in filters:
            match filter[0]:
                case 'exposure':
                    gamma = float(filter[1])

                    height = image.size[0]
                    width = image.size[1]
                    result_img1 = Image.new(mode="RGB", size=(height, width), color=0)
                    for x in range(height):
                        for y in range(width):
                            r = pow(image.getpixel((x, y))[0] / 255, (1 / gamma)) * 255
                            g = pow(image.getpixel((x, y))[1] / 255, (1 / gamma)) * 255
                            b = pow(image.getpixel((x, y))[2] / 255, (1 / gamma)) * 255

                            color = (int(r), int(g), int(b))
                            result_img1.putpixel((x, y), color)

                    image.close()
                    image = result_img1

                case 'saturation':
                    multiplier = float(filter[1])

                    converter = ImageEnhance.Color(image)
                    image = converter.enhance(multiplier)

        t = threading.Thread(target=self.send, args=(image[0], callback_url, self.t))
        t.daemon = True
        t.start()

        return Processed(p, image, p.seed, proc.info)