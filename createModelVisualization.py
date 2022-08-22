import json
import argparse
## import 其他的一些库
import networkVisual
import os
import torch
parser = argparse.ArgumentParser(description='create Feature Visualization.')
parser.add_argument('--config',default="/home/Object_detection/cq/config/modelConfig.json")




class createFeatureVisualization:
    def __init__(self,dataset,model,modelVisualizationConfig,datasetConfig):
        self.dataset = dataset
        self.model = model
        self.modelVisualizationConfig = modelVisualizationConfig
        self.datasetConfig = datasetConfig


    def execAlgo(self):
        for method in self.modelVisualizationConfig["names"]:
            if method == "network":
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if self.model["evaluationObject"] == "UpLoadNet":
                    model = networkVisual.up_load_model_weights(self.model["modelPath"], device,self.model["modelStructurePath"])
                else:
                    model = networkVisual.local_load_model_weights(self.model["modelPath"], device)
                outputPath = networkVisual.transform_to_onnx(device, model,IN_IMAGE_H=self.datasetConfig['size'],IN_IMAGE_W=self.datasetConfig['size'])
                self.json_data = {
                    "modelVisualizationResult":{
                        "VisualizationPath":outputPath,
                        "visualImg":self.output_file_path['dataset']
                    }
                }

    def setSavePath(self,output_file_path):
        # load和save保持一致
        self.loadMethod = self.dataset["loadMethod"]
        self.output_file_path = output_file_path

    def saveLocalFile(self):
        # 本地保存
        save_json = json.dumps(self.json_data, sort_keys=False, indent=2)
        f = open(self.output_file_path['json'], "w")
        f.write(save_json)
        f.close()





if __name__=="__main__":
    # 配置信息来源
    args = parser.parse_args()
    config_path = args.config
    # config_path = "featureConfig.json"
    # 配置信息来源解码
    config_json = json.load(open(config_path, "r"))

    # 解码
    dataset = config_json["dataset"]
    modelVisualizationConfig = config_json["modelVisualizationConfig"]
    model = config_json["model"]
    datasetConfig = config_json["datasetConfig"]
    output_file_path = config_json["outputPath"]

    # 代码运行(结果编码)
    CreateFeatureVisualization = createFeatureVisualization(dataset,model,modelVisualizationConfig,datasetConfig)
    CreateFeatureVisualization.setSavePath(output_file_path)
    CreateFeatureVisualization.execAlgo()
    CreateFeatureVisualization.saveLocalFile()

    # 再次读取保存的文件
    save_json = json.load(open(output_file_path['json'], "r"))
