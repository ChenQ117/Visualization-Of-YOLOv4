import sys
sys.path.append("/home/Object_detection/cq")
from build_model import Build_Model
from eval.evaluator import *
import os.path as osp
import config.yolov4_config as cfg
# coding: utf-8
import os
import numpy as np
import cv2

class CAM:
    """
    dataset：需要可视化的数据集文件夹路径
    model:模型配置文件路径
    """
    def __init__(self,model,modelStructurePath=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if modelStructurePath is None:
            self.model = self.local_load_model_weights(model, self.device)
        else:
            self.model = self.up_load_model_weights(model,self.device,modelStructurePath)
        self.model.eval()


    def exec(self,imgpath,numClass,save_path):
        self.oriPath = imgpath
        img = cv2.imread(self.oriPath)
        img = self.get_img_tensor(img[:, ::-1], 320).to(self.device)
        # print(imgpath,"---------------------------")
        print(img.shape)
        p, p_d, _ = self.model(img)
        p_d = p_d.view(-1, 3, 25)
        print(p_d.shape)
        lists = []
        start = 0
        for x in p:
            end = x.shape[1] * x.shape[1] + start
            lists.append(p_d[start:end:].view(1, x.shape[1], x.shape[1], 3, 25).permute(0, 3, 1, 2, 4))
            start = end
        filename = str(self.oriPath).split("/")[-1]
        class_id = self.getClasses(self.model, self.oriPath, filename)
        print(class_id)
        if len(class_id) == 0:
            print(self.oriPath)
            class_id = np.arange(0, numClass).tolist()
        self.visualizationPath = self.show_CAM(self.oriPath, save_path + '/', filename, lists, class_id)
    def show_CAM(self,image_path,save_path, imgname,feature_maps, class_id, all_ids=25):
        """
        feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
        """
        img_ori = cv2.imread(image_path)
        # img_ori = img_ori[:,:,::-1]
        layers0 = feature_maps[0].reshape([-1, all_ids])#[,85]
        layers1 = feature_maps[1].reshape([-1, all_ids])#[,85]
        layers2 = feature_maps[2].reshape([-1, all_ids])#[,85]
        layers = torch.cat([layers0, layers1, layers2], 0)#[,10]
        #第五个参数
        score_max_v = layers[:, 4].max()
        score_min_v = layers[:, 4].min()

        # for j in range(3):  # layers
        feature = feature_maps[2]
        all_ret = []
        rettt = None
        for id in class_id:
            # 分类的得分
            class_max_v = layers[:, 5 + id].max()
            class_min_v = layers[:, 5 + id].min()
            # compute max of score from three anchor of the layer
            anchors_score_max = feature[0, ..., 4].max(0)[0]
            # compute max of class from three anchor of the layer
            #找出每行的像素点中score最高的值
            anchors_class_max = feature[0, ..., 5 + id].max(0)[0]

            #归一化处理
            scores = ((anchors_score_max - score_min_v) / (
                    score_max_v - score_min_v))

            classes = ((anchors_class_max - class_min_v) / (
                    class_max_v - class_min_v))
            one = scores * classes
            layer_one = one.cpu().detach().numpy()
            #scale处理准备add
            ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            ret = ret.astype(np.uint8)
            gray = ret[:, :, None]
            if rettt is None:
                rettt = ret
            else:
                rettt+=ret
            all_ret.append(gray)

        show = img_ori
        # show = img_ori.copy()
        ret = ((rettt-rettt.min())/(rettt.max()-rettt.min()))*255
        ret = ret.astype(np.uint8)
        # gray = ret[:, :, None]
        ret = cv2.applyColorMap(ret, cv2.COLORMAP_JET)
        ret = cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0]))

        show = show*0.4+ret*0.6
        show = show.astype(np.uint8)
        path =save_path+ imgname
        cv2.imwrite(path,show)
        cv2.waitKey(0)
        print(path)
        return path

    def up_load_model_weights(self,weight_file,device,modelStructurePath):
        """
        加载模型参数
        :param weight_file: 模型权重文件
        :param device: gpu or cpu
        :param modelStructurePath: 模型结构路径
        :return: 加载好了的模型
        """
        import sys
        import importlib
        PROJECT_PATH = osp.abspath(osp.dirname(__file__))
        sys.path.append('/')
        module_py = importlib.import_module(str(modelStructurePath).replace('/','.')[1:]+'.UploadNet')
        showatt = cfg.TRAIN["showatt"]
        chkpt = torch.load(weight_file, map_location=device)
        model = module_py.Build_Model(showatt=showatt).to(device)
        model.load_state_dict(chkpt)
        sys.path.append(PROJECT_PATH)
        print("loading weight file is done")
        del chkpt
        return model
    def local_load_model_weights(self,weight_file,device):
        """
        加载模型参数
        :param weight_file: 模型权重文件
        :param device: gpu or cpu
        :return: 加载好了的模型
        """
        showatt = cfg.TRAIN["showatt"]
        chkpt = torch.load(weight_file, map_location=device)
        model = Build_Model(showatt=showatt).to(device)
        model.load_state_dict(chkpt)
        PROJECT_PATH = osp.abspath(osp.dirname(__file__))
        sys.path.append(PROJECT_PATH)
        print("loading weight file is done")
        del chkpt
        return model
    def get_img_tensor(self,img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(
            img, None
        ).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()

    def getClasses(
            self,model,
                   imgpath,
                   imgname,
                   max_boxes_to_draw=20,
                   min_score_thresh=0.5,
                   ):

        classresult = []
        evalter = Evaluator(model, showatt=cfg.TRAIN["showatt"])
        img = cv2.imread(imgpath)
        bboxes_prd = evalter.get_bbox(img, imgname, mode='det')
        if bboxes_prd.shape[0] != 0:
            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]
            sorted_ind = np.argsort(-scores)  # 得到降序排列的索引
            boxes = boxes[sorted_ind]
            scores = scores[sorted_ind]
            classes = class_inds[sorted_ind]

            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                if scores is None or scores[i] > min_score_thresh:
                    classresult.append(classes[i])
            classresult = list(set(classresult))
            return classresult

if __name__=="__main__":
    import torch
    PROJECT_PATH = osp.abspath(osp.dirname(__file__))
    weight_file = osp.join(PROJECT_PATH, '../weight/best.pt')
    dataset = "/data0/BigPlatform/YOLOv4-pytorch/YOLOv4-cq/data/imgs"
    CAM(dataset,weight_file)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = load_model_weights(weight_file, device)
    # path = "/data0/BigPlatform/YOLOv4-pytorch/YOLOv4-cq/data/imgs/"
    # filename = "000014.jpg"
    # class_id = getClasses(model, cfg.VOC_DATA["CLASSES"], path + filename, filename)
    # # stride = [13, 26, 52]
    #
    # model.eval()
    # img = cv2.imread(path + filename)
    # # img = torch.tensor(img[:,:,-1]).to(device)
    # img = get_img_tensor(img[:, ::-1], 320).to(device)
    # print(img.shape)
    # p, p_d,attan = model(img)
    #
    # p_d = p_d.view(-1,3,25)
    # print(p_d.shape)
    # lists =[]
    # start = 0
    # for x in p:
    #     end = x.shape[1]*x.shape[1]+start
    #     lists.append(p_d[start:end:].view(1,x.shape[1],x.shape[1],3,25).permute(0,3,1,2,4))
    #     start = end
    #
    # save_path = osp.join(PROJECT_PATH, "camResult3")
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # show_CAM(path + filename, save_path + '/', filename, lists, class_id)

