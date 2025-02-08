import json
import re
import h5py
import pickle
import sys

from PIL import Image

MAX_PIXELS = 178956970
RESAMPLING = Image.LANCZOS
def load_image(path_or_url):
    # Lưu lại giá trị giới hạn ban đầu
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    # Tạm thời vô hiệu hóa giới hạn pixel
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path_or_url)

    # Kiểm tra và chuyển đổi ảnh sang RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Kiểm tra số lượng pixel và giảm kích thước nếu cần
    num_pixels = image.width * image.height
    if num_pixels > MAX_PIXELS:
        print(f"Giảm kích thước ảnh từ {image.width}x{image.height} pixels.")
        scaling_factor = (MAX_PIXELS / num_pixels) ** 0.5
        new_width = max(1, int(image.width * scaling_factor))
        new_height = max(1, int(image.height * scaling_factor))
        image = image.resize((new_width, new_height), RESAMPLING)
        print(f"Kích thước ảnh sau khi giảm: {new_width}x{new_height} pixels.")

    # Khôi phục lại giá trị giới hạn ban đầu
    Image.MAX_IMAGE_PIXELS = original_max_pixels
    return image

# Save Pickle
def save_pickle(dic_content, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump(dic_content, file)


# Load Pickle
def load_pickle(path):
    with open(path, 'rb') as file:
        pickle_file = pickle.load(file)
        return pickle_file


# Load h5py File
def load_h5(path):
    with h5py.File(path, 'r') as h5_file:
        h5_data = {}
        for dataset_name in h5_file:
            data = h5_file[dataset_name][:]
            h5_data[dataset_name] = data
        h5_file.close()
        return h5_data
    
            
# Save H5
def save_h5(dic_content, save_path):
    """
        {
            dataset_name1: {
                "dtype": uint32,
                "data": something,
            }
        }
        ...
    """
    with h5py.File(save_path, 'w') as h5_file:
        for dataset_name, item in dic_content.items():
            dtype = item['dtype']
            data = item['data']

            h5_file.create_dataset(
                name=dataset_name,
                dtype=dtype,
                data=data
            )
        h5_file.close()


# Load json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        json_file = json.load(file)
        return json_file
    

# Save json
def save_json(path, dic):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(dic, file, indent=3, ensure_ascii=False)


# Preprocessing data
def clean_text(
        text,
        methods=['rmv_link', 'rmv_punc', 'lower', 'replace_word', 'rmv_space'],
        custom_punctuation = '!"#$%&\'()*+,-:;<=>?@[\\]^_/`{|}~”“',
        patterns=[],
        words_replace=[],
    ):
    cleaned_text = text
    for method in methods:
        if method == 'rmv_link':
            # Remove link
            cleaned_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
            cleaned_text = " ".join(cleaned_text)
        elif method == 'rmv_punc':
            # Remove punctuation
            cleaned_text = re.sub('[%s]' % re.escape(custom_punctuation), '' , cleaned_text)
        elif method == 'lower':
            # Lowercase
            cleaned_text = cleaned_text.lower()
        elif method == 'replace_word':
            # Replace word
            for pattern, repl in zip(patterns, words_replace):
                cleaned_text = re.sub(pattern, repl, cleaned_text)
        elif method == 'rmv_space':
            # Remove extra space
            cleaned_text = re.sub(' +', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
    return cleaned_text
