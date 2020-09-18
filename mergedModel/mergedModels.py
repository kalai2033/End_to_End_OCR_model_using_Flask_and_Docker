import torch
import keras_ocr
import numpy as np
import gc
from flask import Flask, jsonify
from flask_restful import reqparse, Api, Resource
# from flask_restful.utils import cors
from flask_cors import CORS
from data import create_dataset
from models import create_model
from util.visualizer import save_images_return_address
from extracttextfns import loadTextModels, getTextFromGivenImage, clusterBoxes
import werkzeug, os, base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

UPLOAD_FOLDER = 'test1/'

app = Flask(__name__)
api = Api(app)
CORS(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')

torch.set_num_threads(4)
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "test1/")
opt = type('TestOptions', (), {'dataroot': path, 'norm': 'instance', 'input_nc': 1, 'output_nc': 1,
                               'checkpoints_dir': './checkpoints', 'name': 'plates',
                               'netG': 'resnet_9blocks',
                               'preprocess': 'none', 'model': 'test', 'dataset_mode': 'single',
                               'gpu_ids': [],
                               'ngf': 64, 'ndf': 64, 'netD': 'basic', 'n_layers_D': 3,
                               'init_type': 'normal',
                               'init_gain': 0.02, 'no_dropout': False, 'direction': 'AtoB',
                               'serial_batches': True,
                               'num_threads': 0, 'batch_size': 1, 'load_size': 256, 'crop_size': 256,
                               'max_dataset_size': float("inf"), 'no_flip': True, 'display_winsize': 256,
                               'epoch': 'latest', 'load_iter': 0, 'verbose': False, 'suffix': '',
                               'ntest': float("inf"), 'results_dir': './results/', 'aspect_ratio': 1.0,
                               'phase': 'test', 'eval': True, 'num_test': 50, 'model_suffix': '',
                               'isTrain': False,
                               'display_id': -1})()
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)  # regular setup: load and print networks; create schedulers

op_image_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch),
                            'images')  # define the result image directory

# test with eval mode. This only affects layers like batchnorm and dropout.
# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
if opt.eval:
    model.eval()

det_model, recog_model= loadTextModels()
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()
print('Text detection and recognition models are loaded')

print("number of threads used by PyTorch", torch.get_num_threads())


class HealthCheck(Resource):
    def get(self):

        return jsonify(status="Health Status: OK")


class ImageSendForGAN(Resource):
    def post(self):

        data = parser.parse_args()

        if data['file'] == "":
            return {
                'data': '',
                'message': 'No file found',
                'status': 'error'
            }

        photo = data['file']

        if photo:
            # save the image received by POST method
            filename = 'image.jpg'
            photo.save(os.path.join(dirname, UPLOAD_FOLDER + filename))

            # create a dataset given opt.dataset_mode and other options
            dataset = create_dataset(opt)

            for i, data in enumerate(dataset):

                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break

                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths

                if i % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))

                # im = tensor2im(visuals["fake"])
                fake_image_address = save_images_return_address(op_image_dir, visuals, img_path,
                                                                aspect_ratio=opt.aspect_ratio)
                print('Output Image:  ', fake_image_address)
                text, boxes = getTextFromGivenImage(fake_image_address, det_model, recog_model)
                print('Text has been extracted')
                with open(fake_image_address, "rb") as image_file:
                    encoded_bytes = base64.b64encode(image_file.read())
                    encoded_string = encoded_bytes.decode('utf-8')

            torch.cuda.empty_cache()
            dataset = None
            del dataset
            print(gc.collect())

            return {
                'data': text,
                'bboxes': boxes,
                'imageb64': encoded_string,
                'message': 'photo uploaded',
                'status': 'success'
            }

        return {
            'data': '',
            'message': 'Something went wrong',
            'status': 'error'
        }


class ImageSendForOCR(Resource):
    def post(self):

        data = parser.parse_args()

        if data['file'] == "":
            return {
                'data': '',
                'message': 'No file found',
                'status': 'error'
            }

        photo = data['file']

        if photo:
            # save the image received by POST method
            filename = 'image.jpg'
            photo.save(os.path.join(dirname, UPLOAD_FOLDER + filename))

            text, boxes = getTextFromGivenImage(UPLOAD_FOLDER + filename, det_model, recog_model)
            print('Text has been extracted')

            torch.cuda.empty_cache()
            print(gc.collect())

            return {
                'data': text,
                'bboxes': boxes,
                'message': 'photo uploaded',
                'status': 'success'
            }

        return {
            'data': '',
            'message': 'Something went wrong',
            'status': 'error'
        }


class ImageSendForKerasOCR(Resource):
    def post(self):

        data = parser.parse_args()

        if data['file'] == "":
            return {
                'data': '',
                'message': 'No file found',
                'status': 'error'
            }

        photo = data['file']

        if photo:
            # save the image received by POST method
            filename = 'image.jpg'
            photo.save(os.path.join(dirname, UPLOAD_FOLDER + filename))

            image = keras_ocr.tools.read(UPLOAD_FOLDER + filename)
            # Predictions is a list of (text, box) tuples.
            predictions = pipeline.recognize(image=image)
            text = []
            boxes = []
            for t, b in predictions:
                text.append(t)
                boxes.append(b.flatten().tolist())

            clusters = clusterBoxes(boxes)
            listOftextRows = []
            for line in clusters:
                textrowwise = []
                for element in np.asarray(line):
                    textrowwise.append(text[element])
                listOftextRows.append(textrowwise)
            text = listOftextRows

            print('Text has been extracted')

            torch.cuda.empty_cache()
            print(gc.collect())

            return {
                'data': text,
                'bboxes': boxes,
                'message': 'photo uploaded',
                'status': 'success'
            }

        return {
            'data': '',
            'message': 'Something went wrong',
            'status': 'error'
        }


class ImageSendForGANandKerasOCR(Resource):
    def post(self):

        data = parser.parse_args()

        if data['file'] == "":
            return {
                'data': '',
                'message': 'No file found',
                'status': 'error'
            }

        photo = data['file']

        if photo:
            # save the image received by POST method
            filename = 'image.jpg'
            photo.save(os.path.join(dirname, UPLOAD_FOLDER + filename))

            # create a dataset given opt.dataset_mode and other options
            dataset = create_dataset(opt)

            for i, data in enumerate(dataset):

                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break

                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths

                if i % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))

                # im = tensor2im(visuals["fake"])
                fake_image_address = save_images_return_address(op_image_dir, visuals, img_path,
                                                                aspect_ratio=opt.aspect_ratio)
                print('Output Image:  ', fake_image_address)

                image = keras_ocr.tools.read(fake_image_address)
                # Predictions is a list of (text, box) tuples.
                predictions = pipeline.recognize(image=image)
                text = []
                boxes = []
                for t, b in predictions:
                    text.append(t)
                    boxes.append(b.flatten().tolist())
                clusters = clusterBoxes(boxes)
                listOftextRows = []
                for line in clusters:
                    textrowwise = []
                    for element in np.asarray(line):
                        textrowwise.append(text[element])
                    listOftextRows.append(textrowwise)
                text = listOftextRows
                print('Text has been extracted')
                with open(fake_image_address, "rb") as image_file:
                    encoded_bytes = base64.b64encode(image_file.read())
                    encoded_string = encoded_bytes.decode('utf-8')

            torch.cuda.empty_cache()
            dataset = None
            del dataset
            print(gc.collect())

            return {
                'data': text,
                'bboxes': boxes,
                'imageb64': encoded_string,
                'message': 'photo uploaded',
                'status': 'success'
            }

        return {
            'data': '',
            'message': 'Something went wrong',
            'status': 'error'
        }


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(HealthCheck, '/')
api.add_resource(ImageSendForGAN, '/image/')
api.add_resource(ImageSendForOCR, '/imageOCR/')
api.add_resource(ImageSendForKerasOCR, '/imageKerasOCR/')
api.add_resource(ImageSendForGANandKerasOCR, '/imageGANandKerasOCR/')

# if __name__ == '__main__':
#     app.run(port=5000,host='0.0.0.0')
# app.run(debug=True)
