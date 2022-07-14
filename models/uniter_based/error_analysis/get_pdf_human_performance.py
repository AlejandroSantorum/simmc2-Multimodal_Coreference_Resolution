import os
import json
import glob
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF


def set_up_pdf(title=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(185, 5, txt=title, ln=1, align='C')
    pdf.cell(185, 5, txt="", ln=1, align='C') # linebreak
    # Normal font for dialogue history
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0) # black
    return pdf


def print_bboxes(objects, scene_data, draw, outline='red', width=3, offset=0):
    for idx in objects:
        for item in scene_data["scenes"][0]['objects']:
            if item['index'] == idx:
                bboxes = item['bbox']
                # Drawing object bounding box
                draw.rectangle(
                    [(bboxes[0]-offset, bboxes[1]-offset), (bboxes[0]+bboxes[3]+offset , bboxes[1]+bboxes[2]+offset)],
                    outline=outline,
                    width=width
                )
                # Draw IDs with black background
                font = ImageFont.load_default()
                text = str(idx)
                text_width, text_height = font.getsize(text)
                # drawing black rectangle (background) 
                draw.rectangle(
                    (
                        bboxes[0] + offset,
                        bboxes[1] + offset,
                        bboxes[0] + 2 * offset + text_width,
                        bboxes[1] + 2 * offset + text_height,
                    ),
                    fill="black",
                )
                # drawing text (object index)
                draw.text(
                    (bboxes[0]+offset, bboxes[1]+offset), # coordinates
                    text,  # str(object_index)            # text
                    fill=(255, 255, 255),                 # color
                    font=font,                            # font
                )


def add_error_case_pdf(pdf, dialogue, image_for_bboxes, image_plain, ex_num):
    error_img_path_bboxes = "./tmp_error_images/tmp_error_img"+str(ex_num)+"_bboxes.png"
    image_for_bboxes.save(error_img_path_bboxes)
    error_img_path_plain = "./tmp_error_images/tmp_error_img"+str(ex_num)+"_plain.png"
    image_plain.save(error_img_path_plain)

    pdf.add_page()
    pdf.cell(185, 5, txt="DIALOGUE HISTORY of example No. "+str(ex_num), ln=1, align='C')
    pdf.multi_cell(185, 5, txt=dialogue, align='C')

    pdf.image(error_img_path_plain, type='png', w=195, h=100)
    pdf.cell(185, 5, txt="", ln=1, align='C') # linebreak
    pdf.image(error_img_path_bboxes, type='png', w=195, h=100)


def main():
    DATA_PATH = "../processed/random_test_subset/random_devtest_samples.json"
    SCENES_PATH = "../processed/random_test_subset/random_devtest_scenes.json"

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    with open(SCENES_PATH, 'r') as f:
        scenes = json.load(f)
    
    assert len(data) == len(scenes)

    # Setting up FPDF to store
    pdf = set_up_pdf("ESTIMATE HUMAN PERFORMANCE")

    for i in tqdm(range(len(data))):
        objects_ids = data[i]['candidate_ids']
        scene_name = scenes[i]
        if 'm_' in scene_name:
            scene_name = scene_name[2:]
        
        # Getting scene data (scene objects indices and bounding boxes)
        with open("../data/jsons/"+scenes[i]+"_scene.json", 'r') as f:
            scene_data = json.load(f)
        # Getting image
        img_path = "../data/images/" + scene_name + ".png"
        image_for_bboxes = Image.open(img_path)
        draw = ImageDraw.Draw(image_for_bboxes)
        image_plain = Image.open(img_path)
        # Getting dialogue history
        dialogue_history = data[i]['dial'].encode(encoding="latin-1", errors='replace').decode('latin-1')
        # Printing bounding boxes of all items
        print_bboxes(objects_ids, scene_data, draw, outline='white', width=2, offset=0)
        # Adding error case: dialogue + image with boxed items
        add_error_case_pdf(pdf, dialogue_history, image_for_bboxes, image_plain, ex_num=i+1)
    
    # Storing error analysis PDF
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    pdf_store_path = "./pdf_error_analysis/"+"estim_human_perform"+"_"+now_str+".pdf"
    pdf.output(pdf_store_path)

    # deleting auxiliary images
    img_files = glob.glob('./tmp_error_images/*')
    for f in img_files:
        os.remove(f)

        

if __name__ == "__main__":
    main()