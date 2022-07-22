import json
import os
import glob
import argparse
from tqdm import tqdm
import random
from datetime import datetime
from convert_baseline import parse_flattened_results_from_file
from evaluate_only_coref import parse_for_only_coref

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


def get_dialogue_history(example_idx, test_data):
    example = test_data[example_idx]
    dial = example[:example.find(' <SOO>')]

    som_idx = dial.find('<SOM>')
    while som_idx != -1:
        eom_idx = dial.find('<EOM>', som_idx)
        dial = dial[:som_idx] + dial[eom_idx+len('<EOM>'):]
        som_idx = dial.find('<SOM>', som_idx+1)
    return dial


def get_mentioned_objs(example_idx, test_data):
    assert example_idx < len(test_data)

    def _get_mm_context_ids(mm_context):
        ids = []
        start = mm_context.find('<')
        while start != -1:
            end = mm_context.find('>', start)
            id = mm_context[start+1:end]
            ids.append(int(id))
            start = mm_context.find('<', start+1)
        return ids

    example = test_data[example_idx]
    dialogue_part = example[:example.find(' <SOO>')]

    mentioned_ids = []
    som_idx = dialogue_part.find('<SOM>')
    while som_idx != -1:
        eom_idx = dialogue_part.find('<EOM>', som_idx)
        mm_context = dialogue_part[som_idx+len('<SOM>'):eom_idx]
        mentioned_ids += _get_mm_context_ids(mm_context)
        som_idx = dialogue_part.find('<SOM>', som_idx+1)
    return set(mentioned_ids)


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
    model_name = args.model_name
    predictions_file_path = args.predictions_file_path
    target_file_path = args.targets_file_path
    test_examples_file_path = args.test_examples_file_path
    test_scenes_file_path = args.test_scenes_file_path

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(target_file_path)
    list_predicted = parse_for_only_coref(predictions_file_path)
    assert len(list_predicted) == len(list_target)

    # Reading test examples file
    with open(test_examples_file_path, 'r') as f:
        test_data = f.readlines()
    assert len(test_data) == len(list_predicted)

    # Reading scene names files
    with open(test_scenes_file_path, 'r') as f:
        scenes_names = json.load(f)
    assert len(scenes_names) == len(list_predicted)

    # Setting up FPDF to store error cases
    pdf = set_up_pdf()

    checked_examples = set()
    n_errors = 0
    while n_errors < args.n_errors:
        i = random.randint(0, len(list_predicted)-1)
        while i in checked_examples:
            i = random.randint(0, len(list_predicted)-1)
        checked_examples.add(i)

        pred = list_predicted[i][0]['objects']
        target = list_target[i][0]['objects']

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
            dialogue_history = get_dialogue_history(i, test_data).encode(encoding="latin-1", errors='replace').decode('latin-1')
            # Getting mentioned items (<SOM> ... <EOM>)
            mentioned = get_mentioned_objs(i, test_data)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model name
    parser.add_argument('--model_name', default="BART_only_coref")
    # path of the file with the predictions
    parser.add_argument('--predictions_file_path', default="../results/devtest/predictions_input_all_attrs_cp381.txt")
    # path of the file with the test examples
    parser.add_argument('--test_examples_file_path', default="../data_object_special/simmc2_dials_dstc10_devtest_predict.txt")
    # path of the file with the targets
    parser.add_argument('--targets_file_path', default="../data_object_special/simmc2_dials_dstc10_devtest_target.txt")
    # path of the file with the scenes of the used test set
    parser.add_argument('--test_scenes_file_path', default="../data_object_special/simmc2_scenes_devtest.txt")
    # number of random errors to sample
    parser.add_argument('--n_errors', default=50, type=int)

    args = parser.parse_args()

    main(args)
