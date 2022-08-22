import cv2
import torch
import numpy as np
from model.build_model import Build_Model
from eval.evaluator import *
import os.path as osp
import config.yolov4_config as cfg
def show_CAM(image_path,save_path, feature_maps, class_id):
    """
    feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
    """
    img_ori = cv2.imread(image_path)
    img_ori = img_ori[:,:,::-1]
    layers = feature_maps#[,10]
    #第五个参数
    score_max_v = layers[:, 4].max()
    score_min_v = layers[:, 4].min()

    all_ret = []
    for j in class_id:  # layers
        # 分类的得分
        class_max_v = layers[:, 5 + j].max()
        class_min_v = layers[:, 5 + j].min()
        # compute max of score from three anchor of the layer
        anchors_score_max = layers[0, ..., 4].max(0)[0]
        # compute max of class from three anchor of the layer
        # b = layer_one[0, ..., 5 + class_id]
        #找出每行的像素点中score最高的值
        anchors_class_max = layers[0, ..., 5 + j].max(0)[0]

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
        ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        all_ret.append(ret)
    show = img_ori*0.4
    for rets in all_ret:
        rets = cv2.resize(rets, (img_ori.shape[1], img_ori.shape[0]))
        show += rets * 0.2  # 热力图和原图片的比例

    show = show.astype(np.uint8)
    # cv2.imshow(f"one_{SHOW_NAME[idx]}", show)
    path = save_path
    cv2.imwrite(path, show)
    # print(path)
    # print("save111111111111111")
    # cv2.imwrite(save_path + str(j) + 'layer' + str(idx) + SHOW_NAME[idx] + ".jpg", show)
    # cv2.imshow(f"map_{SHOW_NAME[idx]}", ret)


# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = torch.unsqueeze(img,dim=0)

    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

def load_model_weights(weight_file,device):
    """

    :param weight_file: 模型权重文件
    :param device: gpu or cpu
    :return: 加载好了的模型
    """
    showatt = cfg.TRAIN["showatt"]
    chkpt = torch.load(weight_file, map_location=device)
    model = Build_Model(showatt=showatt).to(device)
    model.load_state_dict(chkpt)
    print("loading weight file is done")
    del chkpt
    return model
def get_img_tensor(img, test_shape):
    img = Resize((test_shape, test_shape), correct_box=False)(
        img, None
    ).transpose(2, 0, 1)
    return torch.from_numpy(img[np.newaxis, ...]).float()
if __name__=="__main__":
    import torch
    PROJECT_PATH = osp.abspath(osp.dirname(__file__))
    weight_file = osp.join(PROJECT_PATH, '../weight/best.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_weights(weight_file, device)
    # img_path = "/data0/BigPlatform/YOLOv4-pytorch/YOLOv4-cq/data/imgs/000001.jpg"
    # save_path = osp.join(PROJECT_PATH,'cam.jpg')
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #
    # draw_CAM(model,img_path,save_path,transform=transform)
    # stride = [13, 26, 52]
    path = "/data0/BigPlatform/YOLOv4-pytorch/YOLOv4-cq/data/imgs/"
    imgname = "000014.jpg"
    in_img = torch.randn(1, 3, 416, 416).to(device)
    model.eval()
    img = cv2.imread(path+imgname)
    # img = torch.tensor(img[:,:,-1]).to(device)
    img = get_img_tensor(img[:,::-1],320).to(device)
    p, p_d,attan = model(img)
    #
    # for i in enumerate(p_d):
    #     ret.append(i)
    print(len(p))
    # print(attan.shape)
    p_d = p_d.view(-1,3,25)

    print(p_d.shape)
    save_path = osp.join(PROJECT_PATH, "camResult")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    class_id = [5,6,14]
    show_CAM(path+imgname, save_path + '/' + imgname + "_", p_d,class_id)

