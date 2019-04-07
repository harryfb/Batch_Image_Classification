#=============================================================================
#title           :image_classify.py
#description     :This code will batch classify multiple images  a CSV 
#                 file containing tags using a CNN.
#author          :Harry F Bullough
#date            :10/05/2017
#version         :3.1
#usage           :python image_class.py /my/input/directory/
#notes           :
#python_version  :2.7
#=============================================================================


# Load Libraries
import numpy as np
import tensorflow as tf
import cv2
import csv
import sys
import time
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


modelFile = sys.path[0] + '/output_graph.pb'
labelFile = sys.path[0] + '/output_labels.txt'
inputDirectory = sys.path[0] + '/input/'
outputDirectory = sys.path[0] + '/output/'
imageDirectory = inputDirectory + 'img_'
inputHeight = 299
inputWidth = 299
inputMean = 0
inputStd = 255
inputLayer = 'Placeholder'
outputLayer = 'final_result'


class MyHandler(PatternMatchingEventHandler):
    patterns = ['*img_tags.csv']
    print 'Waiting for change to "img_tags.csv"...'

    def process(self, event):
        process_img()

    def on_modified(self, event):
        print 'img_tags.csv" has been modified'
        self.process(event)


def process_img():
    imageNumber = '0'
    inputFile = open (inputDirectory + 'img_tags.csv','rb')
    imageTags = csv.reader (inputFile)

    csvHeader = ['x','y','img','class']    
    treeFileExists = False
    benchFileExists = False
    binFileExists = False
    undefinedFileExists = False
    slopeFileExists = False


    for row in imageTags:

        # Don't run on the csv header 
        if row[0] != 'x':
            # Read current image number and concatinate to obtain new image path
            imageNumber = row[2]
            imagePath = imageDirectory + imageNumber + '.jpg'

            print 'Processing img_' + imageNumber

            # Classify image using CNN if no slope is detected in image
            if is_slope(imagePath):
                classifier = 'slope'
            else:
                classifier = run_inference_on_image(imagePath)

            # Add new classifier to CSV
            row.append (classifier)

            if classifier == 'tree':
                if not treeFileExists:
                    tree_file = open (outputDirectory + 'trees.csv','wb')
                    tree = csv.writer (tree_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                    tree.writerow (csvHeader)
                tree.writerow (row)

            elif classifier == 'bench':
                if not benchFileExists:
                    bench_file = open (outputDirectory + 'benches.csv','wb')
                    bench = csv.writer (bench_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                    bench.writerow (csvHeader)
                bench.writerow (row)

            elif classifier == 'bin':
                if not binFileExists:
                    bin_file = open(outputDirectory + 'bins.csv','wb')
                    bins = csv.writer(bin_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                    bins.writerow (csvHeader)
                bins.writerow (row)

            elif classifier == 'slope':
                if not slopeFileExists:
                    slope_file = open(outputDirectory + 'slope.csv','wb')
                    slope = csv.writer(slope_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                    slope.writerow(csvHeader)
                slope.writerow (row)

            else:
                if not undefinedFileExists:
                    undefined_file = open(outputDirectory + 'undefined.csv','wb')
                    undefined = csv.writer(undefined_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                    undefined.writerow(csvHeader)
                undefined.writerow (row)

    inputFile.close()
    print 'CSV files generated'

    # OPTIONAL: Send files over SCP to server.

    # if treeFileExists:
    #     tree_file.close()
    #     os.system("sudo scp -i /home/ec2-user/src/ " + outputDirectory + "trees.csv foo@bar.com:/your/directory/trees.csv")
    # if benchFileExists:
    #     bench_file.close()
    #     os.system("sudo scp -i /home/ec2-user/src/ " + outputDirectory + "benches.csv foo@bar.com:/your/directory/benches.csv")
    # if binFileExists:
    #     bin_file.close()
    #     os.system("sudo scp -i /home/ec2-user/src/ " + outputDirectory + "bins.csv foo@bar.com:/your/directory/bins.csv")
    # if undefinedFileExists:
    #     undefined_file.close()
    #     os.system("sudo scp -i /home/ec2-user/src/ " + outputDirectory + "undefined.csv foo@bar.com:/your/directory/undefined.csv")
    # if slopeFileExists:
    #     slope_file.close()
    #     os.system("sudo scp -i /home/ec2-user/src/ " + outputDirectory + "slope.csv foo@bar.com:/your/directory/slope.csv")
        
    print 'Waiting for change to "img_tags.csv"...'

def is_slope(imagePath):
    inputImage = cv2.imread (imagePath, 1)

    # Find image size
    numberOfRows = inputImage.shape[0]
    numberOfColumns = inputImage.shape[1]

    # Define 6 layers to section image
    maskSize = numberOfRows / 6
    maskPosition0 = 0
    maskPosition1 = maskSize - 1
    maskPosition2 = maskPosition1 + maskSize - 1
    maskPosition3 = maskPosition2 + maskSize - 1
    maskPosition4 = maskPosition3 + maskSize - 1
    maskPosition5 = maskPosition4 + maskSize - 1
    maskPosition6 = numberOfRows - 1

    # Crop image into the layers
    layer1 = np.zeros ((maskSize,numberOfColumns))
    layer2 = np.zeros ((maskSize,numberOfColumns))
    layer3 = np.zeros ((maskSize,numberOfColumns))
    layer4 = np.zeros ((maskSize,numberOfColumns))
    layer5 = np.zeros ((maskSize,numberOfColumns))
    layer6 = np.zeros ((maskSize,numberOfColumns))
    layer1 = inputImage[maskPosition0:maskPosition1,:]
    layer2 = inputImage[maskPosition1:maskPosition2,:]
    layer3 = inputImage[maskPosition2:maskPosition3,:]
    layer4 = inputImage[maskPosition3:maskPosition4,:]
    layer5 = inputImage[maskPosition4:maskPosition5,:]
    layer6 = inputImage[maskPosition5:maskPosition6,:]

    # Calculate average colour values in bands
    layer1_b,layer1_g,layer1_r = cv2.split (layer1)
    layer2_b,layer2_g,layer2_r = cv2.split (layer2)
    layer3_b,layer3_g,layer3_r = cv2.split (layer3)
    layer4_b,layer4_g,layer4_r = cv2.split (layer4)
    layer5_b,layer5_g,layer5_r = cv2.split (layer5)
    layer6_b,layer6_g,layer6_r = cv2.split (layer6)

    layer1_b_val = round (cv2.mean (layer1_b)[0])
    layer1_g_val = round (cv2.mean (layer1_g)[0])
    layer1_r_val = round (cv2.mean (layer1_r)[0])
    layer2_b_val = round (cv2.mean (layer2_b)[0])
    layer2_g_val = round (cv2.mean (layer2_g)[0])
    layer2_r_val = round (cv2.mean (layer2_r)[0])
    layer3_b_val = round (cv2.mean (layer3_b)[0])
    layer3_g_val = round (cv2.mean (layer3_g)[0])
    layer3_r_val = round (cv2.mean (layer3_r)[0])
    layer4_b_val = round (cv2.mean (layer4_b)[0])
    layer4_g_val = round (cv2.mean (layer4_g)[0])
    layer4_r_val = round (cv2.mean (layer4_r)[0])
    layer5_b_val = round (cv2.mean (layer5_b)[0])
    layer5_g_val = round (cv2.mean (layer5_g)[0])
    layer5_r_val = round (cv2.mean (layer5_r)[0])
    layer6_b_val = round (cv2.mean (layer6_b)[0])
    layer6_g_val = round (cv2.mean (layer6_g)[0])
    layer6_r_val = round (cv2.mean (layer6_r)[0])

    # Determine if bands are green
    if layer1_g_val > layer1_b_val and layer1_g_val > layer1_r_val:
        layer1Terrain = True
    else:
        layer1Terrain = False
    if layer2_g_val > layer2_b_val and layer2_g_val > layer2_r_val:
        layer2Terrain = True
    else:
        layer2Terrain = False
    if layer3_g_val > layer3_b_val and layer3_g_val > layer3_r_val:
        layer3Terrain = True
    else:
        layer3Terrain = False
    if layer4_g_val > layer4_b_val and layer4_g_val > layer4_r_val:
        layer4Terrain = True
    else:
        layer4Terrain = False
    if layer5_g_val > layer5_b_val and layer5_g_val > layer5_r_val:
        layer5Terrain = True
    else:
        layer5Terrain = False
    if layer6_g_val > layer6_b_val and layer6_g_val > layer6_r_val:
        layer6Terrain = True
    else:
        layer6Terrain = False

    #Set as true if all bands are green or all but the top bands are green
    if layer2Terrain and layer3Terrain and layer4Terrain and layer5Terrain and layer6Terrain:
        return True
    elif layer1Terrain and layer2Terrain and layer3Terrain and layer4Terrain and layer5Terrain and layer6Terrain:
        return True
    else:
        return False


def load_graph(modelFile):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(modelFile, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                inputHeight=299,
                                inputWidth=299,
                                inputMean=0,
                                inputStd=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [inputHeight, inputWidth])
  normalized = tf.divide(tf.subtract(resized, [inputMean]), [inputStd])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(labelFile):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def run_inference_on_image(imagePath):
    answer = "none"

    graph = load_graph(modelFile)
    t = read_tensor_from_image_file(
    imagePath,
    inputHeight=inputHeight,
    inputWidth=inputWidth,
    inputMean=inputMean,
    inputStd=inputStd)

    input_name = "import/" + inputLayer
    output_name = "import/" + outputLayer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })

    results = np.squeeze(results)
    topK = results.argsort()[-5:][::-1]
    labels = load_labels(labelFile)

    answer = labels[topK[0]]
    highScore = results[topK[0]]
    if highScore < 0.5:
      answer = 'undefined'

    return answer


if __name__ == '__main__':
    # Setup observer & assign event handler
    args = sys.argv[1:]
    observer = Observer()
    observer.schedule(MyHandler(), path=args[0] if args else '.')
    observer.start()

    try:
        # Sleep until interrupt detectesd
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
