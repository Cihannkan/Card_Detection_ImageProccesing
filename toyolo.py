import os
import xml.etree.ElementTree as ET

def get_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().strip().split()
    return class_names

def convert_to_yolo_format(xml_file, output_dir, class_name_to_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_name_to_id[class_name]  
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    with open(os.path.join(output_dir, filename.replace('.jpg', '.txt')), 'w') as f:
        f.write('\n'.join(yolo_labels))

class_names_file = r'C:\Users\cihan\OneDrive\Masaüstü\Image\cards.names'
class_names = get_class_names(class_names_file)
class_name_to_id = {name: i for i, name in enumerate(class_names)}

input_dir = r'C:\Users\cihan\OneDrive\Masaüstü\Image\test'  
output_dir = r'C:\Users\cihan\OneDrive\Masaüstü\Image\output'  

os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(input_dir):
    if xml_file.endswith('.xml'):
        convert_to_yolo_format(os.path.join(input_dir, xml_file), output_dir, class_name_to_id)
