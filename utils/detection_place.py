import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torchvision
import sklearn.metrics as sk
from transformers import CLIPTokenizer
from torchvision import datasets
import torch.nn.functional as F
import torchvision
import sys
import json
# from load_tai import train_caption
# from load_tai import Caption_distill_double
# from dassl.utils import load_pretrained_weights
import torchvision.transforms as transforms
from collections import defaultdict

def read_file_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())  # 去除每行末尾的换行符并添加到列表中
    return lines

def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
    elif out_dataset == 'ImageNet10': # the train split is used due to larger and comparable size with ID dataset
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut

def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def input_preprocessing(args, net, images, text_features = None, classifier = None):
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == 'vit-Linear':
        image_features = net(pixel_values = images.float()).last_hidden_state
        image_features = image_features[:, 0, :]
    elif args.model == 'CLIP-Linear':
        image_features = net.encode_image(images).float()
    if classifier:
        outputs = classifier(image_features) / args.T
    else: 
        image_features = image_features/ image_features.norm(dim=-1, keepdim=True) 
        outputs = image_features @ text_features.T / args.T
    
    pseudo_labels = torch.argmax(outputs.detach(), dim=1)
    loss = criterion(outputs, pseudo_labels) # loss is NEGATIVE log likelihood
    loss.backward()

    sign_grad =  torch.ge(images.grad.data, 0) # sign of grad 0 (False) or 1 (True)
    sign_grad = (sign_grad.float() - 0.5) * 2  # convert to -1 or 1

    std=(0.26862954, 0.26130258, 0.27577711) # for CLIP model
    for i in range(3):
        sign_grad[:,i] = sign_grad[:,i]/std[i]

    processed_inputs = images.data  - args.noiseMagnitude * sign_grad # because of nll, here sign_grad is actually: -sign of gradient
    return processed_inputs
  
def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(args.n_cls, args.feat_dim, device =args.gpu)
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            if args.model == 'CLIP': 
                features = net.get_image_features(pixel_values = images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if args.normalize: 
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double()) 
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    torch.save(classwise_mean, os.path.join(args.template_dir,f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join(args.template_dir,f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision

def get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // args.batch_size) and in_dist is False:
                break   
            images, labels = images.cuda(), labels.cuda()
            if args.model == 'CLIP':
                features = net.get_image_features(pixel_values = images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1,1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1,1)), 1)      
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())
        
    return np.asarray(Mahalanobis_score_all, dtype=np.float32)

def get_ood_scores_clip(args, net, loader, test_labels, in_dist=False):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    
    
    test_labels_list=list(test_labels)
    
    # with open('/media/chaod/code/clip_ood/imagenet10_wordnet.json', 'r') as json_file:
    #     data = json.load(json_file)
    # with open('/media/chaod/code/clip_ood/waterbird_wordnet.json', 'r') as json_file:
    #     data = json.load(json_file)
    with open('/media/chaod/code/clip_ood/true/place.json', 'r') as json_file:
        data = json.load(json_file)
    with open('/media/chaod/code/llama-main/clip_ood_datasets/specific_descirbe/full/place_ood_to_describe.json', 'r') as f:
        descriptions = json.load(f)     
    test_labels_list2 = test_labels_list[:]

    
    word_index_dict = {}
    for label in test_labels_list:
        indices_before_extend = len(test_labels_list2)
        # label=label.replace(' ','_')
        # print(label)
        if label in data.keys():
            test_labels_list2.extend(data[label])
        indices_after_extend = len(test_labels_list2)
        word_index_dict[label] = (indices_before_extend , indices_after_extend)
    
    if args.score == 'apd':
        cfg = train_caption.main()
        device= torch.device('cuda:'+str(args.gpu))

        clip_model = Caption_distill_double.load_clip_to_cpu(cfg)
        model = Caption_distill_double.DenseCLIP(cfg, test_labels_list[:500] , clip_model)
        load_pretrained_weights(model.prompt_learner, "/media/chaod/code/TaI-DPT/output/imagenet_10_200_caption/Caption_distill_double/rn50_coco2014/nctx16_cscFalse_ctpend/seed4/prompt_learner/model-best.pth.tar")
        model.to(device)

        clip_model2 = Caption_distill_double.load_clip_to_cpu(cfg)
        model2 = Caption_distill_double.DenseCLIP(cfg, test_labels_list[500:] , clip_model2)
        load_pretrained_weights(model2.prompt_learner, "/media/chaod/code/TaI-DPT/output/imagenet_510_200_caption/Caption_distill_double/rn50_coco2014/nctx16_cscFalse_ctpend/seed4/prompt_learner/model-best.pth.tar")
        model2.to(device)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)
    
    category_counts = defaultdict(int)
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object): 
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            id_class_num = len(test_labels)
            if args.score == 'apd':
                output,_,_,_=model(image=images,captions=None,if_test=True)
                output2,_,_,_=model2(image=images,captions=None,if_test=True)
                output = torch.cat((output,output2),dim=1)
                smax = to_np(F.softmax(output/ args.T, dim=1))
                max_indices = np.argmax(smax,axis=1)
                _score.append(-np.where(max_indices > id_class_num -1, 0 , smax[np.arange(len(smax)), max_indices]))
                # _score.append(-np.max(smax[:id_class_num]*100,axis=1))
                continue
            
            image_features = net.get_image_features(pixel_values = images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if args.model == 'CLIP':
                # text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels_list2], padding=True, return_tensors="pt")
                text_inputs = tokenizer([f"a photo of a {c} {descriptions.get(c,'')[:100]}" for c in test_labels_list2], padding=True, return_tensors="pt")
                text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                                attention_mask = text_inputs['attention_mask'].cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)   
                output = image_features @ text_features.T
            
            
            if args.score == 'max-logit':
                smax = to_np(output)
            else:
                # smax =to_np(output)
                smax = to_np(F.softmax(output/1, dim=1))
            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((args.T*torch.logsumexp(output * 200/ args.T, dim=1))))  #energy score is expected to be smaller for ID
                # _score.append(-to_np((args.T*torch.logsumexp(output / args.T, -1))))
            elif args.score == 'entropy':  
                # raw_value = entropy(smax)
                # filtered = raw_value[raw_value > -1e-5]
                _score.append(entropy(smax, axis = 1)) 
                # _score.append(filtered) 
            elif args.score == 'var':
                _score.append(-np.var(smax, axis = 1))
            elif args.score in ['MCM', 'max-logit']:
                max_indices = np.argmax(smax,axis=1)
                _score.append(-np.max(smax, axis=1)) 
                
                for idx in max_indices:
                    category_counts[test_labels_list2[idx]] += 1
            elif args.score == 'aod':
                max_indices = np.argmax(smax,axis=1)
                _score.append(-np.where(max_indices > id_class_num -1, 0 , smax[np.arange(len(smax)), max_indices]))
                # _score.append(-np.max(smax[:id_class_num],axis=1))
            elif args.score == 'aneard':
                max_indices = np.argmax(to_np(output),axis=1) #判断ID以外的OOD类是否有相似度更大的
                max_values = np.max(to_np(output[:,:100]),axis=1)
                _score.append(-np.where(max_indices > id_class_num -1, max_values*0.7 , max_values))#如果有，就放缩域内最大的相似度
            elif args.score == 'changed-softmax':
                
                num_rows, _ = output.size()
                for i in range(num_rows):
                    max_scores = []
                    output_row = output[i]
                    for i, label in enumerate(test_labels_list):
                        ID_value = output_row[i]                                                                                                                                                                            
                        word_indices_range = word_index_dict[label]
                        if word_indices_range[0]==word_indices_range[1]:
                            max_scores.append(ID_value)
                            continue
                        max_index = torch.argmax(output_row[word_indices_range[0]:word_indices_range[1]]) + word_indices_range[0]
                        max_values, _ = torch.max(torch.stack([output_row[max_index], ID_value]), dim=0)
                        max_scores.append(max_values)
                    max_scores = torch.tensor(max_scores)
                    values = to_np(torch.exp(output_row)/torch.sum(torch.exp(max_scores)))
                    max_indices = np.argmax(values)
                    if max_indices < id_class_num:
                        max_values = -np.max(values)
                    else:
                        max_values = -np.max(values) * 0.95
                    _score.append(max_values) 
            elif args.score == 'min-max': # k个二分类思想，对每个ID, 找到类别簇里的最大值是否是OOD类别，如果是OOD最大，就判定为OOD。取所有ID簇里的最大值，作为置信度
                num_rows, _ = output.size()
                for i in range(num_rows):
                    max_scores = []
                    output_row = output[i]
                    
                    for i, label in enumerate(test_labels_list):
                        
                        ID_value = output_row[i]   #获取当前id class的logit   
                                                                                                                                                                          
                        word_indices_range = word_index_dict[label] #获取near ood下标的范围

                        if word_indices_range[0]==word_indices_range[1]:
                            max_scores.append(ID_value) #如果没有，就append ID
                            continue
                        max_index = torch.argmax(output_row[word_indices_range[0]:word_indices_range[1]]) + word_indices_range[0]
                        max_values, max_index_id = torch.max(torch.stack([output_row[max_index], ID_value]), dim=0)
                        
                        if max_index_id.item() == 0 :
                            ID_value = ID_value * 0.8 # 判断最大值是否是OOD，如果是，就将OOD乘以0.8
                            
                        
                        max_scores.append(ID_value)

                    max_scores = torch.tensor(max_scores)
                    values = to_np(max_scores)
                    if args.in_dataset == 'ImageNet20':
                        T = 0.06
                    else:
                        T = 1
                    # values = to_np(torch.exp(max_scores/T)/torch.sum(torch.exp(max_scores/T)))
                    
                    max_index = np.argmax(values)
                    category_counts[test_labels_list2[max_index]] += 1
                    _score.append(-np.max(values)) 
            elif args.score == 'max-min': # 先找到最大的ID类在哪里，然后判断该类别的类别簇中的OOD是否比ID更大，如果有，则认为是OOD，取ID的0.8
                num_rows, _ = output.size()
                
                for i in range(num_rows):
                    
                    max_scores = []
                    output_row = output[i]
                    max_index_within_test_labels = torch.argmax(output_row[:len(test_labels_list)])
                    label = test_labels_list[max_index_within_test_labels]
                    
                    word_indices_range = word_index_dict[label]
                    ID_value = output_row[max_index_within_test_labels]
                    
                    if word_indices_range[0]==word_indices_range[1]:
                        max_scores.append(ID_value) #如果没有，就append ID
                        
                    else:
                        max_index = torch.argmax(output_row[word_indices_range[0]:word_indices_range[1]]) + word_indices_range[0]
                        max_values, max_index_id = torch.max(torch.stack([output_row[max_index], ID_value]), dim=0)
                        
                        if max_index_id.item() == 0 and max_values - ID_value > 0.01:
                            ID_value = ID_value * 0.8 # 判断最大值是否是OOD，如果是，就将OOD乘以0.8
                        max_scores.append(ID_value)
                    
                    # values = to_np(max_scores)
                    device= torch.device('cuda:'+str(args.gpu))
                    max_scores = torch.tensor(max_scores).to(device)
                    output_row = torch.tensor(output_row[:len(test_labels_list)]).to(device)
                    values = to_np(torch.exp(max_scores)/torch.sum(torch.exp(output_row)))
             
                    _score.append(-np.max(values)) 
    
    print("MCM_describe")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        print(f"{category}: {count}")
    
    if args.score == 'changed-softmax' or args.score == 'min-max' or args.score == 'max-min':
        return _score.copy()   
     
    return concat(_score)[:len(loader.dataset)].copy()   

def get_id_scores_clip(args, net, loader, test_labels, in_dist=False):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object): 
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            id_class_num = len(test_labels)
            # print(test_labels)
            test_labels_list=list(test_labels)
            image_features = net.get_image_features(pixel_values = images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if args.model == 'CLIP':
                text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels_list], padding=True, return_tensors="pt")
                text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                                attention_mask = text_inputs['attention_mask'].cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)   
                output = image_features @ text_features.T
            if args.score == 'max-logit':
                smax = to_np(output)
            else:
                smax = to_np(F.softmax(output/ args.T, dim=1))
            if args.score == 'energy':
                #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
                _score.append(-to_np((1.2*torch.logsumexp(output / args.T, dim=1))))  #energy score is expected to be smaller for ID
            elif args.score == 'entropy':  
                # raw_value = entropy(smax)
                # filtered = raw_value[raw_value > -1e-5]
                _score.append(entropy(smax, axis = 1)) 
                # _score.append(filtered) 
            elif args.score == 'var':
                _score.append(-np.var(smax, axis = 1))
            elif args.score in ['MCM', 'max-logit']:
                _score.append(-np.max(smax, axis=1)) 
            elif args.score == 'aod':
                max_indices = np.argmax(smax,axis=1)
                _score.append(-np.where(max_indices > id_class_num , 0 , smax[np.arange(len(smax)), max_indices]))
                
    return concat(_score)[:len(loader.dataset)].copy()

def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)

class TextDataset(torch.utils.data.Dataset):
    '''
    used for MIPC score. wrap up the list of captions as Dataset to enable batch processing
    '''
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Load data and get label
        X = self.texts[index]
        y = self.labels[index]

        return X, y
