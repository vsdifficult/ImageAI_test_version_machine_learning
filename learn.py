

from imageai.Classification.Custom import ClassificationModelTrainer 


model_tr = ClassificationModelTrainer() 
model_tr.setModelTypeAsMobileNetV2() 
model_tr.setModelTypeAsResNet50()
model_tr.setModelTypeAsInceptionV3()  
model_tr.setModelTypeAsDenseNet121() 

model_tr.setDataDirectory(r'./Custom datasets/people') 
model_tr.trainModel(num_experiments=100, batch_size=32) 
