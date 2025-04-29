from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\YOLOv12\VehicleDetection_YOLOv12\exp15\weights\best.pt')
    model.predict(source=r'/workspace/Highway_17_2020-07-30_jpg.rf.8fee29e029bb32554fe14d1e81dfcf12.jpg',
                  save=True,
                  show=False,
                  ) 