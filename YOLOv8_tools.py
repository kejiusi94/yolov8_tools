import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from xml.dom.minidom import Document
import xml.etree.ElementTree as et
import tqdm
import os
from ultralytics import YOLO


class YOLOv8:
    def __init__(self, images_path, xml_path, txt_path, model_path, device_id, img_type, conf, iou, size, num_classes, task):
        self.images_path = images_path
        self.xml_path = xml_path
        self.txt_path = txt_path
        self.model_path = model_path
        self.device_id = device_id
        self.img_type = img_type
        self.conf = conf
        self.iou = iou
        self.size = size
        self.num_classes = num_classes
        self.task = task
        if self.model_path.split(".")[1] == "onnx":
            self.session, self.model_inputs, self.input_shape = self.model_init()
        self.img_list = self.get_image_path()

    def get_image_path(self):
        """
        获取文件夹下所有图片的路径。

        返回值：
        - image_list: 一个列表，包含文件夹内的所有图片。
        """
        image_list = []
        for img_path in os.listdir(self.images_path):
            if img_path.endswith(tuple(self.img_type)):
                image_list.append(img_path)
        return image_list

    def img_preprocess(self, img_path):
        img = cv2.imread(img_path)
        img_copy = img.copy()
        h, w, c = img_copy.shape[:3]
        # 将图像颜色空间从BGR转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将图像大小调整为匹配输入形状
        img = cv2.resize(img, (self.size, self.size))
        # 通过除以255.0来归一化图像数据
        img_data = np.array(img) / 255.0
        # 转置图像，使通道维度为第一维
        img_data = np.transpose(img_data, (2, 0, 1))  # 通道首
        # 扩展图像数据的维度以匹配预期的输入形状
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

        return img_data, h, w, c

    def model_init(self):
        if self.model_path.split(".")[1] == "onnx":
            if torch.cuda.is_available():
                print("Using CUDA")
                providers = [('CUDAExecutionProvider', {'device_id': self.device_id})]
            else:
                print("Using CPU")
                providers = ["CPUExecutionProvider"]
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(self.model_path,
                                           session_options=session_options,
                                           providers=providers)  # 加载模型对话

            model_inputs = session.get_inputs()
            input_shape = model_inputs[0].shape
            return session, model_inputs, input_shape
        else:
            return 0

    def nms_calculate_iou(self, boxes, scores):
        """

        :param boxes: 边界框
        :param scores: 得分

        :return: 边界框索引
        """
        # 如果没有边界框，则直接返回空列表
        if len(boxes) == 0:
            return []
        # 将得分和边界框转换为NumPy数组
        scores = np.array(scores)
        boxes = np.array(boxes)
        # 根据置信度阈值过滤边界框
        mask = scores > self.conf
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        # 如果过滤后没有边界框，则返回空列表
        if len(filtered_boxes) == 0:
            return []
        # 根据置信度得分对边界框进行排序
        sorted_indices = np.argsort(filtered_scores)[::-1]
        # 初始化一个空列表来存储选择的边界框索引
        index = []
        # 当还有未处理的边界框时，循环继续
        while len(sorted_indices) > 0:
            # 选择得分最高的边界框索引
            current_index = sorted_indices[0]
            index.append(current_index)
            # 如果只剩一个边界框，则结束循环
            if len(sorted_indices) == 1:
                break
            # 获取当前边界框和其他边界框
            current_box = filtered_boxes[current_index]
            other_boxes = filtered_boxes[sorted_indices[1:]]

            # 计算当前边界框与其他边界框的IoU
            x1 = np.maximum(current_box[0], np.array(other_boxes)[:, 0])
            y1 = np.maximum(current_box[1], np.array(other_boxes)[:, 1])

            x2 = np.minimum(current_box[0] + current_box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
            y2 = np.minimum(current_box[1] + current_box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
            # 计算交集区域的面积
            intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            # 计算给定边界框的面积
            box_area = current_box[2] * current_box[3]
            # 计算其他边界框的面积
            other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
            # 计算IoU值
            iou = intersection_area / (box_area + other_boxes_area - intersection_area)

            # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
            non_overlapping_indices = np.where(iou <= self.iou)[0]
            # 更新sorted_indices以仅包含不重叠的边界框
            sorted_indices = sorted_indices[non_overlapping_indices + 1]
        # 返回选择的边界框索引
        return index

    def onnx_model(self):
        time1 = time.time()
        error_list = []
        for index in tqdm.tqdm(range(len(self.img_list)), desc='检测进度'):
            img_name = self.img_list[index]
            img_path = os.path.join(self.images_path, img_name)
            img_data, h, w, c = self.img_preprocess(img_path)
            output = self.session.run(None, {self.model_inputs[0].name: img_data})
            # 转置和压缩输出以匹配预期的形状
            outputs = np.transpose(np.squeeze(output[0]))
            # 获取输出数组的行数
            rows = outputs.shape[0]
            # 用于存储检测的边界框、得分和类别ID的列表
            boxes = []
            scores = []
            class_ids = []
            # 计算边界框坐标的缩放因子
            x_factor = w / self.input_shape[2]
            y_factor = h / self.input_shape[3]
            # 遍历输出数组的每一行
            for i in range(rows):
                # 从当前行提取类别得分
                classes_scores = outputs[i][4:]
                # 找到类别得分中的最大得分
                max_score = np.amax(classes_scores)
                # 如果最大得分高于置信度阈值
                if max_score >= self.conf:
                    # 获取得分最高的类别ID
                    class_id = np.argmax(classes_scores)
                    # 从当前行提取边界框坐标
                    box_x, box_y, box_w, box_h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                    # 计算边界框的缩放坐标
                    left = int((box_x - box_w / 2) * x_factor)
                    top = int((box_y - box_h / 2) * y_factor)
                    width = int(box_w * x_factor)
                    height = int(box_h * y_factor)
                    # 将类别ID、得分和框坐标添加到各自的列表中
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])
            # 应用非最大抑制过滤重叠的边界框
            indices = self.nms_calculate_iou(boxes, scores)
            # 遍历非最大抑制后的选定索引
            all_class_id = []
            xml_boxs = []
            for i in indices:
                # 根据索引获取框、得分和类别ID
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                all_class_id.append(class_id)
                if len(box) == 0:
                    error_list.append(os.path.join(self.images_path, img_path))
                    continue
                xml_box = [int(box[0]), int(box[1]), int((box[0] + box[2])), int((box[1] + box[3]))]
                xml_boxs.append(xml_box)
                self.to_xml(img_name, xml_boxs, w, h, c, all_class_id)
        time2 = time.time()
        if len(error_list) == 0:
            print(f"已检测完毕，用时{time2 - time1}s")
        else:
            print('\n'.join(error_list))
            print(f"以上图片未检测到目标，共{len(error_list)}张")

    def to_xml(self, img_name, xml_boxs, w, h, c, all_class_id):

        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        folder = xmlBuilder.createElement("folder")  # folder标签
        folder_content = xmlBuilder.createTextNode("dataname")
        folder.appendChild(folder_content)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filename_content = xmlBuilder.createTextNode(img_name[0:-4] + ".jpg")
        filename.appendChild(filename_content)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        width_content = xmlBuilder.createTextNode(str(w))
        width.appendChild(width_content)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        height_content = xmlBuilder.createTextNode(str(h))
        height.appendChild(height_content)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depth_content = xmlBuilder.createTextNode(str(c))
        depth.appendChild(depth_content)
        size.appendChild(depth)  # size子标签depth结束
        annotation.appendChild(size)  # size标签结束

        for x in range(len(all_class_id)):
            object = xmlBuilder.createElement("object")  # object 标签
            pic_name = xmlBuilder.createElement("name")  # name标签
            name_content = xmlBuilder.createTextNode(self.num_classes[int(all_class_id[x])])
            pic_name.appendChild(name_content)
            object.appendChild(pic_name)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            pose_content = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(pose_content)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncated_Content = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncated_Content)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficult_content = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficult_content)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            xminContent = xmlBuilder.createTextNode(str(xml_boxs[x][0]))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            yminContent = xmlBuilder.createTextNode(str(xml_boxs[x][1]))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            xmaxContent = xmlBuilder.createTextNode(str(xml_boxs[x][2]))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            ymaxContent = xmlBuilder.createTextNode(str(xml_boxs[x][3]))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束
            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束
        f = open(os.path.join(self.xml_path, img_name.split(".")[0] + ".xml"), 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()

    def pt_model(self):
        time1 = time.time()
        error_list = []
        model = YOLO(self.model_path)
        if torch.cuda.is_available() and self.device_id >= 0:
            device = torch.device('cuda:' + str(self.device_id))
            print("Using CUDA")
        else:
            device = torch.device('cpu')
            print("Using cpu")
        model.to(device)
        all_class_id = []
        xml_boxs = []
        for i in tqdm.tqdm(range(len(self.img_list)), desc='检测进度'):
            img_name = self.img_list[i]
            img_path = os.path.join(self.images_path, img_name)
            img_data, h, w, c = self.img_preprocess(img_path)
            result = model(img_path)
            # 获取检测结果中的边界框
            boxes = result[0].boxes.xyxy.cpu().numpy()  # xyxy格式的边界框坐标
            if len(boxes) == 0:
                error_list.append(os.path.join(self.images_path, img_path))
                continue
            xml_box = [round(boxes[0][0]), round(boxes[0][1]), round(boxes[0][2]), round(boxes[0][3])]
            xml_boxs.append(xml_box)
            all_class_id = result[0].boxes.cls.cpu().numpy()  # 检测到的类别编号
            self.to_xml(img_name, xml_boxs, w, h, c, all_class_id)
            time2 = time.time()
            if len(error_list) == 0:
                print(f"已检测完毕，用时{time2 - time1}s")
            else:
                print('\n'.join(error_list))
                print(f"以上图片未检测到目标，共{len(error_list)}张")

    def txt2xml(self):
        files = os.listdir(self.txt_path)
        for i, name in enumerate(files):
            img_name = name[0:-3]+self.img_type
            img_path = os.path.join(self.images_path, img_name)
            txt_file = open(os.path.join(self.txt_path, name))
            txt_list = txt_file.readlines()
            img_data, h, w, c = self.img_preprocess(img_path)
            all_class_id = []
            xml_boxs = []
            for j in txt_list:
                oneline = j.strip().split(" ")
                x1 = int(((float(oneline[1])) * w + 1) - (float(oneline[3])) * 0.5 * w)
                y1 = int(((float(oneline[2])) * h + 1) - (float(oneline[4])) * 0.5 * h)
                x2 = int(((float(oneline[1])) * w + 1) + (float(oneline[3])) * 0.5 * w)
                y2 = int(((float(oneline[2])) * h + 1) + (float(oneline[4])) * 0.5 * h)
                xml_box = [x1, y1, x2, y2]
                all_class_id.append(oneline[0])
                xml_boxs.append(xml_box)
            self.to_xml(img_name, xml_boxs, w, h, c, all_class_id)

    def xml2txt(self):
        file_names = []
        for root, dirs, files in os.walk(self.xml_path):
            for file in files:
                if os.path.splitext(file)[1] == '.xml':
                    f = os.path.splitext(file)[0]
                    file_names.append(f)
        for i in file_names:
            xml_name = os.path.join(self.xml_path, i+".xml")
            txt_result = ''
            outfile = open(xml_name, encoding='UTF-8')
            filetree = et.parse(outfile)
            outfile.close()
            root = filetree.getroot()

            # 获取图片大小
            pic_size = root.find('size')
            w = int(pic_size.find('width').text)
            h = int(pic_size.find('height').text)

            for obj in root.findall('object'):
                # 获取类别名
                class_name = obj.find('name').text
                for k, v in self.num_classes.items():
                    if v== class_name:
                        class_id = k
                # 获取每个obj的bbox框的左上和右下坐标
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                dw = 1. / w
                dh = 1. / h
                txt_x = ((x1 + x2) / 2.0) * dw
                txt_y = ((y1 + y2) / 2.0) * dh
                txt_w = (x2 - x1) * dw
                txt_h = (y2 - y1) * dh
                txt = '{} {} {} {} {}\n'.format(class_id, txt_x, txt_y, txt_w, txt_h)
                txt_result = txt_result + txt

            f = open(os.path.join(self.txt_path, i + ".txt"), 'a')
            f.write(txt_result)
            f.close()

    def main(self):
        if self.model_path.split(".")[1] == "pt":
            print(f"你的模型后缀为.pt,生成的xml文件将存入{self.xml_path}")
            self.pt_model()
        elif self.model_path.split(".")[1] == "onnx":
            print(f"你的模型后缀为.onnx,生成的xml文件将存入{self.xml_path}")
            self.onnx_model()
        elif self.task == 'xml2txt':
            print(f"任务为xml转txt,转换的文件将存入{self.txt_path}")
            self.xml2txt()
        elif self.task == 'txt2xml':
            print(f"任务为txt转xml,转换的文件将存入{self.xml_path}")
            self.txt2xml()
        else:
            print("请确保正确选择了任务{model2xml, xml2txt, txt2xml}")
            print("其中模型后缀支持pt/onnx")
            return KeyError


if __name__ == "__main__":

    # 创建用于处理命令行参数的解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", type=str, default='./images', help="输入图像文件夹的路径.")
    parser.add_argument("-xml", type=str, default='./xml', help="输出xml文件夹的路径.")
    parser.add_argument("-txt", type=str, default='./txt', help="输出txt文件夹的路径.")
    parser.add_argument("-model", type=str, default='model/yolov8n.onnx', help="请输入您的ONNX模型路径.")
    parser.add_argument("-device", type=int, default=0, help="GPU序号.")
    parser.add_argument("-type", type=str, default='jpg', help="待检测图片类别.")
    parser.add_argument("-conf", type=float, default=0.35, help="置信度阈值.")
    parser.add_argument("-iou", type=float, default=0.45, help="iou阈值.")
    parser.add_argument("-size", type=int, default=640, help="模型的输入尺寸.")
    parser.add_argument("-classes", type=dict, default={0: "person"}, help="所有类别的字典.")
    parser.add_argument("-task", type=str, default='xml2txt', help="任务模式，可选任务{model2xml, xml2txt, txt2xml}.")

    args = parser.parse_args()

    # 创建YOLOv8实例
    detection = YOLOv8(args.img, args.xml, args.txt, args.model, args.device, args.type, args.conf, args.iou, args.size,
                       args.classes, args.task)
    detection.main()
