# WARNING: 'simmc2_scene_jsons_dstc10_public.zip' folder and 'simmc2_scene_images_dstc10_public_part2.zip'
#          folders have to be unziped into 'public' folder and 'simmc2_scene_images_dstc10_public_part2' folders
#          respectively

SCENES="cloth_store_paul_19_3 cloth_store_paul_19_5 cloth_store_paul_18_2 cloth_store_paul_18_4 cloth_store_paul_18_5" 

python3 visualize_bboxes.py --screenshot_root ../../data/simmc2_scene_images_dstc10_public_part2 \
                            --scene_json_root ../../data/public \
                            --save_root ./visualized_bboxes \
                            --scene_names $SCENES