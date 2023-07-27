import os

class Config:
    def __init__(self, args):
        if args.dataset_file == 'H2O':
            self.dataset = args.dataset_file
            self.IR_dim = 308
            self.num_frames = 64
            self.num_obj_classes = 11
            self.num_action_classes = 37
            self.hand_idx = [9, 10]
            self.obj2idx = {
                "book" : 1,
                "espresso" : 2,
                "lotion" : 3,
                "lotion_spray" : 4,
                "milk" : 5,
                "cocoa" : 6,
                "chips" : 7,
                "cappuccino" : 8,
            }
            self.action2idx = {
                'grab book':1,
                'grab espresso':2,
                'grab lotion':3,
                'grab spray':4,
                'grab milk':5,
                'grab cocoa':6,
                'grab chips':7,
                'grab cappuccino':8,
                'place book':9,
                'place espresso':10,
                'place lotion':11,
                'place spray':12,
                'place milk':13,
                'place cocoa':14,
                'place chips':15,
                'place cappuccino':16,
                'open lotion':17,
                'open milk':18,
                'open chips':19,
                'close lotion':20,
                'close milk':21,
                'close chips':22,
                'pour milk':23,
                'take out espresso':24,
                'take out cocoa':25,
                'take out chips':26,
                'take out cappuccino':27,
                'put in espresso':28,
                'put in cocoa':29,
                'put in cappuccino':30,
                'apply lotion':31,
                'apply spray':32,
                'read book':33,
                'read espresso':34,
                'spray spray':35,
                'squeeze lotion':36}   
            self.cam_param = [636.6593017578125, 636.251953125, 635.283881879317, 366.8740353496978, 1280, 720]
            self.object_model_path = os.path.join(args.data_path, 'H2O/object')
                
        elif args.dataset_file == 'FPHA':
            self.dataset = args.dataset_file
            self.IR_dim = 204
            self.num_frames = 32
            self.num_obj_classes = 6
            self.num_action_classes = 11
            self.hand_idx = [5]
            self.obj2idx = {
                "juice_bottle" : 1,
                "liquid_soap" : 2,
                "milk" : 3,
                "salt" : 4,
            }
            self.action2idx = {
                'open_juice_bottle': 1, 
                'close_juice_bottle': 2, 
                'pour_juice_bottle': 3, 
                'open_milk': 4, 
                'close_milk': 5, 
                'pour_milk': 6, 
                'put_salt': 7, 
                'open_liquid_soap': 8, 
                'close_liquid_soap': 9, 
                'pour_liquid_soap': 10}
            self.cam_param = [1395.749023, 1395.749268, 935.732544, 540.681030, 1920, 1080]
            self.object_model_path = os.path.join(args.data_path, 'FPHA/Object_models')