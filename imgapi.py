import requests
import io
import base64
from PIL import Image, ImageTk
import tkinter as tk
from threading import Thread
import time

url = "http://127.0.0.1:7860"

with open('\\image1.jpeg',"rb") as f:
    global openposeimage
    openposeimage = base64.b64encode(f.read()).decode()

loramodel = ""
prompt = f'''solo,beautiful woman,exquisite facial features, realistic facial details,full frontal, smile, indoor, masterpiece, official art, 8K, CG rendering, complex details, (photography: 1.1), (gorgeous Chinese woman: 1.3), Instagram, (photo realism: 0.8)'''

negative_prompt=f'''canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render,((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), (((tranny))), (((trans))), (((trannsexual))), (hermaphrodite), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). (((more than 2 nipples))). [[[adult]]], out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))'''


payload = {
  "prompt": prompt+f",{'<lora:'+loramodel+':1>' if loramodel != '' else ''}",
  "negative_prompt": negative_prompt,
  "steps": 35,
  "alwayson_scripts": {
    "controlnet": {
      "args": [
        {
          "input_image": openposeimage,
          "module": "openpose",
          "model": "control_v11p_sd15_openpose_fp16 [73c2b67d]"
        }
      ]
    },
    "ADetailer": {
      "args": [
        {
          "ad_model": "face_yolov8n.pt",
          "ad_prompt": "(((blue eyes)))",
          "ad_negative_prompt": "",
          "ad_confidence": 1,
          "ad_mask_k_largest": 0,
          "ad_mask_min_ratio": 0.0,
          "ad_mask_max_ratio": 1.0,
          "ad_dilate_erode": 32,
          "ad_x_offset": 0,
          "ad_y_offset": 0,
          "ad_mask_merge_invert": "None",
          "ad_mask_blur": 4,
          "ad_denoising_strength": 0.4,
          "ad_inpaint_only_masked": 1,
          "ad_inpaint_only_masked_padding": 0,
          "ad_use_inpaint_width_height": 0,
          "ad_inpaint_width": 512,
          "ad_inpaint_height": 512,
          "ad_use_steps": 1,
          "ad_steps": 28,
          "ad_use_cfg_scale": 0,
          "ad_cfg_scale": 7.0,
          "ad_use_sampler": 0,
          "ad_sampler": "DPM++ 2M Karras",
          "ad_use_noise_multiplier": 0,
          "ad_noise_multiplier": 1.0,
          "ad_use_clip_skip": 0,
          "ad_clip_skip": 1,
          "ad_restore_face": 0,
          "ad_controlnet_model": "None",
          "ad_controlnet_module": "None",
          "ad_controlnet_weight": 1.0,
          "ad_controlnet_guidance_start": 0.0,
          "ad_controlnet_guidance_end": 1.0
        }
      ]
    }
  }
}

isDone = False

def getProgress():
    global isDone
    while True:
      if isDone == True:
          break
      time.sleep(0.5)
      response1 = requests.get(f'{url}/sdapi/v1/progress')
      print(response1.json()['progress'])


if __name__ == '__main__':
  t = Thread(target=getProgress)
  t.start()
  response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
  r = response.json()
  isDone = True
  k=1
  for i in r['images']:
      image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
      image.save(f'\\output{k}.png')
      k+=1
  print('生成完毕')
  window= tk.Tk()
  window.title('生成结果')
  image = Image.open('\\output1.png')
  image = ImageTk.PhotoImage(image)
  label = tk.Label(window, image=image)
  label.pack()
  window.mainloop()
