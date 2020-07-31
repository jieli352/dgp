import numpy as np
import PIL
from IPython import display
from matplotlib.cm import get_cmap

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.ontology_pb2 import Ontology
from dgp.utils.protobuf import open_pbobject
from dgp.utils.visualization import visualize_semantic_segmentation_2d, visualize_instance_segmentation_2d
import cv2
import os
plasma_color_map = get_cmap('plasma')

# Define high level variables
DDAD_TEST_JSON_PATH = '/data/datasets/ddad/ddad_test_all_labels/ddad_test.json'
TEMP_OUTPUT = './'

ddad_test = SynchronizedSceneDataset(
    DDAD_TEST_JSON_PATH,
    split='test',
    datum_names=('CAMERA_01',),
    requested_annotations=('semantic_segmentation_2d', 'instance_segmentation_2d'),
    only_annotated_datums=True
    #generate_depth_from_datum='lidar'
)
print('Loaded DDAD train split containing {} samples'.format(len(ddad_test)))

random_sample_idx = np.random.randint(len(ddad_test))
sample = ddad_test[random_sample_idx] # scene[0] - lidar, scene[1:] - camera datums
sample_datum_names = [datum['datum_name'] for datum in sample]
print('Loaded sample {} with datums {}'.format(random_sample_idx, sample_datum_names))


# visualize 2D instance masks. 
image = np.array(sample[0]['rgb'])
cv2.imwrite(os.path.join(TEMP_OUTPUT, "raw.jpg"), image)

semseg_ontology = open_pbobject(ddad_test.scenes[0].ontology_files['semantic_segmentation_2d'], Ontology)
instance_ontology = open_pbobject(ddad_test.scenes[0].ontology_files['instance_segmentation_2d'], Ontology)

semantic_segmentation_2d_annotation = sample[0]['semantic_segmentation_2d']
sem_seg = visualize_semantic_segmentation_2d(
    semantic_segmentation_2d_annotation, semseg_ontology, image=image, debug=False
)
cv2.imwrite(os.path.join(TEMP_OUTPUT, "segmentation.jpg"), sem_seg)
mask_list = sample[0]['panoptic_instance_masks']
class_ids = sample[0]['panoptic_class_ids']
class_names = sample[0]['panoptic_class_names']
ins_seg = visualize_instance_segmentation_2d(mask_list, class_ids,
    instance_ontology, image.shape[:2], class_names=class_names, image=image, white_edge=True)
cv2.imwrite(os.path.join(TEMP_OUTPUT, "instance.jpg"), ins_seg)
       