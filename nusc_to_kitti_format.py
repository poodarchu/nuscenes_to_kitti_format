import os
import numpy as np
import random

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from PIL import Image
from pyquaternion import Quaternion

import shutil

nusc = NuScenes(
    version='v1.0-mini',
    dataroot='/home/zhubenjin/data/nuScenes_full',
    verbose=True)

cnt = 0

scenes = nusc.scene
random.shuffle(scenes)

train_scenes = [scene['token'] for scene in scenes[:5]]
val_scenes = [scene['token'] for scene in scenes[5:]]

detection_to_general = {
    'ignore': [
        'animal',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.stroller',
        'human.pedestrian.wheelchair',
        'movable_object.debris',
        'movable_object.pushable_pullable',
        'static_object.bicycle_rack',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
    ],
    'barrier': [
        'movable_object.barrier',
    ],
    'bicycle': [
        'vehicle.bicycle',
    ],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'car': [
        'vehicle.car',
    ],
    'construction_vehicle': [
        'vehicle.construction',
    ],
    'motorcycle': ['vehicle.motorcycle'],
    'pedestrian': [
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
    ],
    'traffic_cone': ['movable_object.trafficcone'],
    'trailer': ['vehicle.trailer'],
    'truck': [
        'vehicle.truck',
    ],
}

general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore', 
}


for sample in nusc.sample:
    cam_token = sample['data']['CAM_FRONT']
    pc_token = sample['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', pc_token)
    # pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])

    cam = nusc.get('sample_data', cam_token)
    # print(cam)

    # im = Image.open(osp.join(nusc.dataroot, cam['filename']))
    # print(im.width, im.height)

    data_path, box_list, cam_intrinsic = nusc.get_sample_data(pc_token)

    # print(data_path)
    # print(pcl_path)

    # fname =  pcl_path.split('/')[-1].split('.')[0] + '.txt'
    # print(fname)

    # if sample['scene_token'] in train_scenes
    if sample['scene_token'] in val_scenes:
        fout = open(
            osp.join(os.getcwd(), 'validation', 'label_2', pc_token + '.txt'),
            'w')

        # pc = LidarPointCloud.from_file(pcl_path)

        # print(pc.points.shape)
        # print(len(box_list))
        # print(dir(box_list[0]))

        dontcares = []
        for box in box_list:
            w, l, h = box.wlh
            x, y, z = box.center
            z = z - h / 2

            # r = Quaternion(axis=[0, 0, 1], angle=-3.14159265/2)
            # yaw = box.orientation.rotate(r).yaw_pitch_roll[0]
            yaw, pitch, roll = box.orientation.yaw_pitch_roll
            yaw = -yaw - np.pi / 2
            if yaw < -np.pi:
                yaw += np.pi * 2

            name = general_to_detection[box.name]
            line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                name, -1, -1, -1, -1, -1, -1, -1, h, w, l, -y, -z, x, yaw)
            if name == 'ignore':
                dontcares.append(line)
            else:
                fout.write(line)

        for line in dontcares:
            fout.write(line)

        fout.close()

        # img = Image.new('RGB', (im.width, im.height), (255, 255, 255))
        # img.save(osp.join(os.getcwd(), 'KITTI_image', cam['filename'].split('/')[-1].split('.')[0] + '.png'), "PNG")

        shutil.copy(
            data_path,
            os.path.join(nusc.dataroot, 'validation', 'velodyne',
                         pc_token + '.bin'))
        shutil.copy(
            osp.join(nusc.dataroot, cam['filename']),
            os.path.join(nusc.dataroot, 'validation', 'image_2',
                         pc_token + '.png'))
        shutil.copy(
            osp.join(nusc.dataroot, 'dummy.txt'),
            osp.join(nusc.dataroot, 'validation', 'calib_2', pc_token + '.txt'))

        cnt += 1
    
print(cnt)
