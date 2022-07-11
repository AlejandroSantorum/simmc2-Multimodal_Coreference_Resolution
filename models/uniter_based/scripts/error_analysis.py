import json
import os
import glob
import argparse
from tqdm import tqdm
import random
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF


def set_up_pdf(title=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(185, 5, txt=title, ln=1, align='C')
    pdf.cell(185, 5, txt="", ln=1, align='C') # linebreak
    # Addint color legend
    pdf.set_text_color(255, 255, 255) # sort of light green
    pdf.cell(185, 5, txt="· MENTIONED OBJECT IDs", ln=1, align='C', fill=True)
    pdf.set_text_color(41, 255, 69) # sort of light green
    pdf.cell(185, 5, txt="· PREDICTED OBJECT IDs", ln=1, align='C', fill=True)
    pdf.set_text_color(255, 51, 0) # sort of light red
    pdf.cell(185, 5, txt="· TARGET OBJECT IDs", ln=1, align='C', fill=True)
    # Normal font for dialogue history
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0) # black
    return pdf


def parse_objects_from_simmc2_format(data):
    objects_list = []
    for dial in data['dialogue_data']:
        for turn in dial['dialogue']:
            try:
                if turn['disambiguation_label'] == 1:
                    continue
            except:
                pass
            objs = turn['transcript_annotated']['act_attributes']['objects']
            objects_list.append(objs)
    return objects_list


def get_mentioned_objs(test_data_entry):
    candidate_ids = test_data_entry['candidate_ids']
    mentioned = test_data_entry['candidate_mentioned']
    return [id for k,id in enumerate(candidate_ids) if mentioned[k]==1]


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
                try:
                    font = ImageFont.truetype("Arial.ttf", size=25)
                except:
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


def add_error_case_pdf(pdf, dialogue, image, ex_num):
    error_img_path = "./tmp_error_images/tmp_error_img"+str(ex_num)+".png"
    image.save(error_img_path)

    pdf.add_page()
    pdf.cell(185, 5, txt="DIALOGUE HISTORY of example No. "+str(ex_num), ln=1, align='C')
    pdf.multi_cell(185, 5, txt=dialogue, align='C')

    pdf.image(error_img_path, type='png', w=195, h=100)



def main(args):
    model_name = "UNITER_objmen_noIDs_devtest"
    predictions_file_path = "../output/eval_UNITER_basic_all_objmen_noIDs_devtest.json"
    target_file_path = "../data/simmc2_dials_dstc10_devtest.json"
    test_examples_file_path = "../processed/devtest.json"
    test_scenes_file_path = "../processed/simmc2_scenes_devtest.txt"

    # Reading predictions file and parsing it
    with open(predictions_file_path, 'r') as f:
        pred_data = json.load(f)
    list_predicted = parse_objects_from_simmc2_format(pred_data)
    
    # Reading target file and parsing it
    with open(target_file_path, 'r') as f:
        target_data = json.load(f)
    list_target = parse_objects_from_simmc2_format(target_data)

    assert len(list_predicted) == len(list_target)

    # Reading test examples file
    with open(test_examples_file_path, 'r') as f:
        test_data = json.load(f)
    assert len(test_data) == len(list_predicted)

    # Reading scene names files
    with open(test_scenes_file_path, 'r') as f:
        scenes_names = json.load(f)
    assert len(scenes_names) == len(list_predicted)

    # Setting up FPDF to store error cases
    pdf = set_up_pdf(model_name)

    checked_examples = set()
    n_errors = 0
    while n_errors < args.n_errors:
        i = random.randint(0, len(list_predicted)-1)
        while i in checked_examples:
            i = random.randint(0, len(list_predicted)-1)
        checked_examples.add(i)

        pred = list_predicted[i]
        target = list_target[i]

        if (len(pred) != len(target)) or (len(pred) != len(set(pred).intersection(target))):
            n_errors += 1
            print("Analyzing example no.", i+1, "("+str(n_errors)+"/"+str(args.n_errors)+")")
            if 'm_' in scenes_names[i]:
                scene_name = scenes_names[i][2:]
            else:
                scene_name = scenes_names[i]
            
            # Getting scene data (scene objects indices and bounding boxes)
            with open("../data/jsons/"+scenes_names[i]+"_scene.json", 'r') as f:
                scene_data = json.load(f)
            # Getting image
            img_path = "../data/images/" + scene_name + ".png"
            image = Image.open(img_path)
            draw = ImageDraw.Draw(image)
            # Getting dialogue history
            dialogue_history = test_data[i]['dial'].encode(encoding="latin-1", errors='replace').decode('latin-1')
            # Getting mentioned items (multimodal context)
            mentioned = get_mentioned_objs(test_data[i])
            # Printing predicted items, target items and mentioned items in the dialogue
            print_bboxes(pred, scene_data, draw, outline='#29ff45', width=4, offset=0)
            print_bboxes(target, scene_data, draw, outline='#ff3300', width=4, offset=4)
            print_bboxes(mentioned, scene_data, draw, outline='white', width=3, offset=7)
            # Adding error case: dialogue + image with boxed items
            add_error_case_pdf(pdf, dialogue_history, image, ex_num=i+1)
    
    # Storing error analysis PDF
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
    pdf_store_path = "./pdf_error_analysis/"+model_name+"_"+now_str+".pdf"
    pdf.output(pdf_store_path)

    # deleting auxiliary images
    img_files = glob.glob('./tmp_error_images/*')
    for f in img_files:
        os.remove(f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_errors', default=50, type=int)
    parser.add_argument('--error_imgs_file', default="", type=str) # TODO
    args = parser.parse_args()

    main(args)