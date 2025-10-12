import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from torch import Tensor, dtype

from rope import Models
from rope import VideoManager

@dataclass
class FoundSourceFace:
    face_kps: ndarray
    face_emb: ndarray
    cropped_img: Tensor

models = Models.Models()

load_file = open("saved_parameters.json", "r")
parameters = json.load(load_file)
load_file.close()

vm = VideoManager.VideoManager(models)
vm.parameters = parameters

vm.control = {'MaskViewButton': False}

import logging
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_source_faces(source_image: ndarray) -> list[FoundSourceFace]:

    img: Tensor = torch.from_numpy(source_image).to('cuda')
    img = img.permute(2,0,1)
    kpss: ndarray[Any, dtype[np.floating[_64Bit]]] = models.run_detect(img, max_num=50)

    ret: list[FoundSourceFace] = []
    for face_kps in kpss:

        face_emb, cropped_img = models.run_recognize(img, face_kps)
        ret.append(FoundSourceFace(face_kps, face_emb, cropped_img))

    return ret

def main():

    dirs_config_file = open("batch_data.json", "r")
    dirs_parameters: dict = json.load(dirs_config_file)
    dirs_config_file.close()

    target_face_embedding: ndarray[Any, dtype[Any]] = get_target_face_embedding(dirs_parameters["target_face_path"])
    torch.cuda.empty_cache()

    source_directory = dirs_parameters["input_dir"]
    input_filenames: list[str] = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, input_filenames)
        in os.walk(source_directory)
        for f in input_filenames
    ]

    total_input_files = len(input_filenames)
    logger.info('total input files: ' + str(total_input_files))

    output_directory = dirs_parameters["output_dir"]
    exist_files = [f for (_, _, filenames) in os.walk(output_directory) for f in filenames]

    for i, input_file in enumerate(input_filenames):

        target_media_file_name = input_file.replace('\\', '/')

        path_components = target_media_file_name.split('/')
        filename = path_components[-1].split('.')[0]
        dir_name = path_components[-2]
        target_filename = dir_name + '_' + filename + ".jpg"

        if target_filename in exist_files:
            logger.info('file ' + target_filename + ' already exists, skipping')
            continue

        logger.info(
            f'---------- processing {i+1}th file: ' + input_file + f' out of total {total_input_files} files --------------'
        )

        source_image: ndarray = cv2.imread(input_file) # BGR
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB) # RGB
        found_source_faces: list[FoundSourceFace] = find_source_faces(source_image)

        face_params: list[dict[str, Any]] = [
            {
                "Embedding": foud_source_face.face_emb,
                "SourceFaceAssignments": [0],
                "EmbeddingNumber": 0,  #used for adding additional found faces
                'AssignedEmbedding': target_face_embedding,  #the currently assigned source embedding
            }
            for foud_source_face in found_source_faces
        ]
        vm.assign_found_faces(face_params)
        output_img: ndarray = vm.swap_video(source_image, 0, False)

        saved_filename = os.path.join(output_directory, target_filename)
        cv2.imwrite(saved_filename, cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        logger.debug('write file: ' + saved_filename)

        torch.cuda.empty_cache()
        logger.info('---------- processing file ' + input_file + ' completed ----------------')


def get_target_face_embedding(target_face_img_path: str) -> ndarray:
    target_face_img: ndarray = cv2.imread(target_face_img_path)

    img: Tensor = torch.from_numpy(target_face_img.astype('uint8')).to('cuda')

    pad_scale = 0.2
    padded_width = int(img.size()[1] * (1. + pad_scale))
    padded_height = int(img.size()[0] * (1. + pad_scale))

    padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device='cuda:0')

    width_start = int(img.size()[1] * pad_scale / 2)
    width_end = width_start + int(img.size()[1])
    height_start = int(img.size()[0] * pad_scale / 2)
    height_end = height_start + int(img.size()[0])

    padding[height_start:height_end, width_start:width_end, :] = img
    img = padding

    img = img.permute(2, 0, 1)

    kpss: ndarray = models.run_detect(img, max_num=1)[0]

    target_face_emb, target_face_cropped_image = models.run_recognize(img, kpss)
    return target_face_emb

if __name__ == '__main__':
    main()