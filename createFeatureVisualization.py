import json
import argparse
import os
import os.path as osp
import sys
sys.path.append('/home/Object_detection/cq/model')
## import 其他的一些库
import CAMVisual3

parser = argparse.ArgumentParser(description='create Feature Visualization.')
parser.add_argument('--config',default="/home/Object_detection/cq/config/featureConfig.json")






class createFeatureVisualization:
    def __init__(self,dataset,model,featureVisualizationConfig,datasetConfig):
        self.dataset = dataset
        self.model = model
        self.featureVisualizationConfig = featureVisualizationConfig
        self.datasetConfig = datasetConfig
        self.json_data = []


    def execAlgo(self):
        for method in self.featureVisualizationConfig["names"]:
            if method == "CAM":
                if self.model["evaluationObject"] == "UpLoadNet":
                    cam = CAMVisual3.CAM(self.model["modelPath"],self.model["modelStructurePath"])
                else:
                    cam = CAMVisual3.CAM(self.model["modelPath"])
                save_dataset = self.output_file_path["dataset"]
                if not os.path.exists(save_dataset):
                    os.makedirs(save_dataset)
                filenames = os.listdir(self.dataset["path"])
                for filename in filenames:
                    imgpath = osp.join(self.dataset["path"], filename)
                    cam.exec(imgpath,self.datasetConfig["numClass"], save_dataset)
                    data = {
                        "featureVisualizationResult": {
                            "OriPath": cam.oriPath,
                            "VisualizationPath": cam.visualizationPath,
                        }
                    }
                    self.json_data.append(data)

    def setSavePath(self,output_file_path):
        # load和save保持一致
        self.loadMethod = self.dataset["loadMethod"]
        self.output_file_path = output_file_path

    def saveLocalFile(self):
        # 本地保存
        save_json = json.dumps(self.json_data, sort_keys=False, indent=2)
        f = open(self.output_file_path["json"], "w")
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
    visualMethod = config_json["featureVisualizationConfig"]
    model = config_json["model"]
    output_file_path = config_json["outputPath"]
    datasetConfig = config_json["datasetConfig"]
    # 代码运行(结果编码)
    CreateFeatureVisualization = createFeatureVisualization(dataset,model,visualMethod,datasetConfig)
    CreateFeatureVisualization.setSavePath(output_file_path)
    CreateFeatureVisualization.execAlgo()
    CreateFeatureVisualization.saveLocalFile()

    # 再次读取保存的文件
    save_json = json.load(open(output_file_path["json"], "r"))
