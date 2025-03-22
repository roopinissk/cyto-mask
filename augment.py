import augly.image as imaugs
import os
import random
import glob
from tqdm import tqdm
from PIL import Image
import shutil  # Added missing import

# Set random seed for reproducibility
random.seed(0)

# Define augmentation functions

# def saturate(img_path):
#     with Image.open(img_path) as img:
#         if img.mode == 'L':
#             head, tail = os.path.split(img_path)
#             new_img_path = os.path.join(head, f'saturated_skip___' + tail)
#             shutil.copy2(img_path, new_img_path)
#             return new_img_path
        
#     random_saturation_factor = 0.03 - random.random()
#     head, tail = os.path.split(img_path)
#     new_img_path = os.path.join(head, f'saturated_{random_saturation_factor:.2f}___' + tail)
#     imaugs.saturation(image=img_path, output_path=new_img_path, factor=random_saturation_factor)
#     return new_img_path

# def add_noise(img_path):
#     noise_mean = random.uniform(0, 0.05)
#     head, tail = os.path.split(img_path)
#     new_img_path = os.path.join(head, f'noise_{noise_mean:.2f}_0___' + tail)
#     imaugs.random_noise(image=img_path, output_path=new_img_path, mean=noise_mean, seed=0)
#     return new_img_path

# def brighten(img_path):
#     img = Image.open(img_path)
#     if img.mode != 'RGB':
#         rgb_img = Image.new('RGB', img.size)
#         rgb_img.paste(img)
#         temp_path = img_path + '_temp_rgb.png'
#         rgb_img.save(temp_path)
#         img_path = temp_path
    
#     random_brightness_factor = random.uniform(0.8, 1.2)

#     head, tail = os.path.split(img_path)
#     new_img_path = os.path.join(head, f'brightness_{random_brightness_factor:.2f}___' + os.path.basename(tail))
#     imaugs.brightness(image=img_path, output_path=new_img_path, factor=random_brightness_factor)
    
#     if img.mode != 'RGB' and os.path.exists(temp_path):
#         os.remove(temp_path)
    
#     return new_img_path



def rotate(img_path, output_dir):
    random_degree = random.randrange(45, 305)
    base = os.path.basename(img_path)
    base_name, _ = os.path.splitext(base)
    new_filename = f"{base_name}_rotated{random_degree}_.png"
    new_img_path = os.path.join(output_dir, new_filename)
    
    # Most libraries have an 'expand' parameter to prevent cropping
    imaugs.rotate(image=img_path, output_path=new_img_path, degrees=random_degree)
    return new_img_path

    

def augment_brightfield_images(bf_dir, output_dir="dataset/augment"):
    os.makedirs(output_dir, exist_ok=True)
    bf_images = glob.glob(os.path.join(bf_dir, "*.png"))
    print(f"Found {len(bf_images)} brightfield images in {bf_dir}")
    
    for img_path in tqdm(bf_images, desc="Processing images"):
        # Save directly to output_dir
        rotate(img_path, output_dir)
        
# Call the function
augment_brightfield_images("./dataset/bf", output_dir="dataset/augment")


# def identity(_):
#     pass

# def factor_4_point_5_aug(img_path):
#     task_list = [
#         saturate,
#         rotate,
#         rotate,
#         add_noise,
#     ]
#     random.shuffle(task_list)
#     if random.random() < 0.5:
#         task_list = task_list[:-1]
#     task_list += [identity]
#     for task in task_list:
#         task(img_path)


# def factor_1_point_5_aug(img_path):
#     task_list = [
#         saturate,
#         rotate,
#         add_noise,
#     ]
#     random.shuffle(task_list)
#     task_list = task_list[:-2]
#     if random.random() < 0.5:
#         task_list = task_list[:-1]
#     task_list += [identity]
#     for task in task_list:
#         task(img_path)

# def factor_7_aug(img_path):
#     task_list = [
#         saturate,
#         saturate,
#         rotate,
#         rotate,
#         add_noise,
#         add_noise,
#         identity,
#     ]
#     for task in task_list:
#         task(img_path)

# def sample_only(img_path, n):
#     head, tail = os.path.split(img_path)
#     total_images = len(os.listdir(head))
#     if n >= total_images:
#         return
#     discarded_folder = os.path.join(head, 'discarded')
#     if not os.path.isdir(discarded_folder):
#         try:
#             os.mkdir(discarded_folder)
#         except Exception as e:
#             raise e
    
    
#     if random.random() > (n/total_images):
#         src_path = img_path
#         dst_path = os.path.join(discarded_folder, tail)
#         shutil.move(src_path, dst_path)


# transform_func = {
#     'dust': (sample_only, [3000]),
#     'degraded': (factor_4_point_5_aug, []),
#     'big_halo': (factor_7_aug, []),
#     'small_halo': (factor_1_point_5_aug, []),
#     'medium_halo': (factor_4_point_5_aug, [])
# }

# ## run this from inside labelled_sorted
# def main():
#     here_path = os.getcwd()
#     print("here", here_path)
#     for dir in os.listdir(here_path):
#         image_dir = os.path.join(here_path, dir)
#         print("here", dir, image_dir)
#         if not os.path.isdir(image_dir):
#             continue
#         print("dir", dir, image_dir)
#         for image_file in os.listdir(image_dir):
#             image_file_path = os.path.join(image_dir, image_file)
#             if not os.path.isfile(image_file_path):
#                 continue
#             print("image_file", image_file, image_file_path)
#             transform_func[dir][0](image_file_path, *transform_func[dir][1])    
    

# if __name__ == '__main__':
#     main()