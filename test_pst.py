import os, argparse, time, datetime, sys, shutil, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.PST_dataset import PST_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
from model import SpiderMesh

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='SpiderMesh')
parser.add_argument('--weight_name', '-w', type=str, default='SpiderMesh_152')
parser.add_argument('--file_name', '-f', type=str, default='pst900.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=720) 
parser.add_argument('--img_width', '-iw', type=int, default=1280)  
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=5)
parser.add_argument('--n_layer', '-nl', type=int, default=152)
parser.add_argument('--data_dir', '-dr', type=str, default='./pst_dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
args = parser.parse_args()
#############################################################################################
 
if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save direcotry
    if os.path.exists("./runs"):
        print("previous \"./runs\" folder exist, will delete this folder")
        shutil.rmtree("./runs")
    os.makedirs("./runs")
    os.makedirs("./runs/vis_results")

    model_dir = os.path.join(args.model_dir)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the loaded model file.')
    else:
        sys.exit('no model file found.') 
    print('testing %s-%s on GPU #%d with pytorch' % (args.model_name, args.n_layer, args.gpu))
    
    conf_total_rgb = np.zeros((args.n_class, args.n_class))
    conf_total_thermal = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(num_classes=args.n_class, num_layers=args.n_layer)
    if args.gpu >= 0: model.cuda(args.gpu)

    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)  
    print('done!')

    batch_size = 1
    test_dataset  = PST_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False)

    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            logits_rgb, logits_thermal = model(images)
            end_time = time.time()
            if it>=5: ave_time_cost += (end_time-start_time)

            label = labels.cpu().numpy().squeeze().flatten()
            prediction_rgb = logits_rgb.argmax(1).cpu().numpy().squeeze().flatten()
            prediction_thermal = logits_thermal.argmax(1).cpu().numpy().squeeze().flatten()

            # generate confusion matrix frame-by-frame
            conf_rgb = confusion_matrix(y_true=label, y_pred=prediction_rgb, labels=[0,1,2,3,4]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total_rgb += conf_rgb
            conf_thermal = confusion_matrix(y_true=label, y_pred=prediction_thermal, labels=[0,1,2,3,4]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total_thermal += conf_thermal
            # save demo images
            # visualize(image_name=names, predictions=logits_rgb.argmax(1), weight_name=args.weight_name, name='rgb')
            # visualize(image_name=names, predictions=logits_thermal.argmax(1), weight_name=args.weight_name, name='thermal')
            # visualize(image_name=names, predictions=torch.tensor(label).cuda().reshape(1,720,1280), weight_name=args.weight_name, name='gt')

            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
 
    precision_per_class_rgb, recall_per_class_rgb, iou_per_class_rgb = compute_results(conf_total_rgb)
    precision_per_class_thermal, recall_per_class_thermal, iou_per_class_thermal = compute_results(conf_total_thermal)

    conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    savemat(conf_total_matfile,  {'conf': conf_total_rgb})
 
    print('\n###########################################################################')
    print('\n%s-%s test results (with batch size %d) on %s using %s:' %(args.model_name, args.n_layer, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu))) 
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width)) 
    print('* the weight name: %s' %args.weight_name) 
    print('* the file name: %s\n' %args.file_name) 

    print("################################ RGB ##################################")
    print("* iou per class: \n  unlabeled: %.6f, hand_drill: %.6f, backpack: %.6f, fire_extinguisher: %.6f, survivor: %.6f" \
          %(iou_per_class_rgb[0], iou_per_class_rgb[1], iou_per_class_rgb[2], iou_per_class_rgb[3], iou_per_class_rgb[4])) 
    print("* miou: %.6f\n" %(np.mean(np.nan_to_num(iou_per_class_rgb))))

    print("################################ Thermal ##################################")
    print("* iou per class: \n  unlabeled: %.6f, hand_drill: %.6f, backpack: %.6f, fire_extinguisher: %.6f, survivor: %.6f" \
          %(iou_per_class_thermal[0], iou_per_class_thermal[1], iou_per_class_thermal[2], iou_per_class_thermal[3], iou_per_class_thermal[4])) 
    print("* miou: %.6f" %(iou_per_class_thermal.mean()))
    print('\n###########################################################################')
