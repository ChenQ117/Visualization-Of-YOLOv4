
import sys
sys.path.append('/home/Object_detection/cq/model')
# from tool.utils import *
from model.build_model import Build_Model
from eval.evaluator import *
import config.yolov4_config as cfg
import os.path as osp

def transform_to_onnx(device,model,batch_size= 1, IN_IMAGE_H= 416, IN_IMAGE_W= 416):
    """
    device:cpu or gpu
    model:加载好了的模型
    IN_IMAGE_H:图片的高度
    IN_IMAGE_W:图片的宽度
    返回模型保存的路径
    """

    input_names = ["input"]
    output_names = ['boxes', 'confs']
    PROJECT_PATH = osp.abspath(osp.dirname(__file__))
    dynamic = False
    if batch_size <= 0:
        dynamic = True
    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).to(device)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return PROJECT_PATH+'/'+onnx_file_name
    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).to(device)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          )

        print('Onnx model exporting done')
        return PROJECT_PATH+'/'+onnx_file_name


def up_load_model_weights(weight_file, device, modelStructurePath):
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
    module_py = importlib.import_module(str(modelStructurePath).replace('/', '.')[1:]+'.UploadNet')
    showatt = cfg.TRAIN["showatt"]
    chkpt = torch.load(weight_file, map_location=device)
    model = module_py.Build_Model(showatt=showatt).to(device)
    model.load_state_dict(chkpt,False)
    sys.path.append(PROJECT_PATH)
    print("loading weight file is done")
    del chkpt
    return model


def local_load_model_weights(weight_file, device):
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
    sys.path.append(PROJECT_PATH)
    print("loading weight file is done")
    del chkpt
    return model


# def change_model(weight_file):
#     print("开始保存。。。。")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     showatt = cfg.TRAIN["showatt"]
#     model = Build_Model(showatt=showatt).to(device)
#     model = load_model_weights(weight_file, model, device)
#
#     height = 416
#     width = 416
#     channel = 3
#     batch = 1
#     input = torch.randn(batch, channel, height, width).to(device)
#     print("input", input.shape)
#     # 导出为onnx格式
#     onnx_path = "onnx_model.onnx"
#     torch.onnx.export(model, input, onnx_path)
#     # trace_model = torch.jit.trace(self.__model, input)
#     # trace_model.save("mtrace.pt")
#     # print("my.pt")
#     # torch.save(self.__model,"my.pt_"+time.time())
#
#     print("保存结束了。。。。")

if __name__ == '__main__':
    import os.path as osp
    PROJECT_PATH = osp.abspath(osp.dirname(__file__))
    weight_file = osp.join(PROJECT_PATH, '../weight/best.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = local_load_model_weights(weight_file, device)
    modelStructurePath = '/data0/BigPlatform/YOLOv4-pytorch/YOLOv4-cq/model'
    model = up_load_model_weights(weight_file,device,modelStructurePath)
    a = transform_to_onnx(device,model)
    # change_model(weight_file)