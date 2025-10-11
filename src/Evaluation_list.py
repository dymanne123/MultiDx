import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
import collections
from utlis import *
import argparse
import re

# Uncomment the line below to download the necessary NLTK data for the first-time use
# import nltk
# nltk.download('punkt')




def parse_args():
    parser = argparse.ArgumentParser()
    
    return parser.parse_args()


def calculate_EM(a_gold, a_pred):
    """Calculate Exact Match (EM) score"""
    return a_gold.replace(' ', '').lower() == a_pred.replace(' ', '').lower()


def calculate_F1(a_gold, a_pred):
    """Calculate token-level F1 score"""
    a_gold=a_gold.lower()
    a_pred=a_pred.lower()
    gold_toks = word_tokenize(a_gold)
    pred_toks = word_tokenize(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), 0,0
    if num_same == 0:
        return 0,0,0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall), precision,  recall


def parse_generation(pred):
    """Parse generated answers based on rules"""
    for start_id in ['<answer>']:
        if start_id in pred:
            pred = pred.split(start_id)[-1].strip()
            break
    for end_id in ['</answer>']:
        if end_id in pred:
            pred = pred.split(end_id)[0].strip()
            break
    #print(pred)
    diagnoses = []
    for line in pred.strip().splitlines():
        #line = line.strip()
        match = re.match(r'^\d+\.\s*(.+)', line)
        if match:
            diag = match.group(1).strip().lower()
            diagnoses.append(diag)
    if diagnoses==[]:
        diagnoses.append(pred.strip().lower())
    return diagnoses    


def initialize_metrics(num_question_cat):
    """Initialize dictionaries to store metrics"""
    EM_dict = [0, 0] 
    f1_score_dict = []
    return EM_dict, f1_score_dict


def process_file(data):
    """Process a single file for predictions and metrics calculation"""
    pred = data['prediction'].strip()
    
    preds = parse_generation(pred)
    
    gts = data['final_diagnosis']
    #gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]
    return preds, gts


def compute_metrics(folder_path, dataset_name,IND_list,hit):
    """Compute EM and F1 metrics for all files"""
    num_question_cat = 9 if dataset_name == 'TGQA' else 1
    EM_dict, f1_score_dict = initialize_metrics(num_question_cat)
    precision,recall=0,0
    num_test_samples = 300
    acc=[]
    for i in range(num_test_samples):
        file_path = folder_path + f'/{str(i)}.json'
        if not os.path.exists(file_path):
            continue
        #if i not in IND_list:
        #    continue
        with open(file_path) as json_file:
            data = json.load(json_file)
        try:
            preds, gts = process_file(data)
        except Exception as e:
            print(i)
            continue
        #print(preds, gts)
        if preds is None:
            print(i)
            continue
                
        max_f1 = 0.0
        max_precision = 0.0
        max_recall = 0.0
        max_EM = 0.0
        num_result=0
        for pred in preds:
            num_result+=1
            if num_result > hit:
                break
            cur_f1_score, cur_precision, cur_recall = calculate_F1(pred, gts) 
            cur_EM = calculate_EM(pred, gts)
            if cur_f1_score > max_f1:
                max_f1 = cur_f1_score
            if cur_precision > max_precision:
                max_precision = cur_precision
            if cur_recall > max_recall:
                max_recall = cur_recall
            if cur_EM > max_EM:
                max_EM = cur_EM
        f1_score_dict.append(max_f1)
        EM_dict[0] += max_EM
        EM_dict[1] += 1
        precision+=max_precision
        recall+=max_recall
        if max_EM==1:
            acc.append(i)
    print(acc)
        #print(i)


    return EM_dict, f1_score_dict, precision/len(f1_score_dict), recall/len(f1_score_dict)


def print_results(EM_dict, f1_score_dict,precision,recall):
    """Print final EM and F1 results"""
    
    if EM_dict[1] > 0:
        EM_dict[0] = EM_dict[0]/EM_dict[1]
    print('\nEM:')
    print(EM_dict[0], 
         EM_dict[1])
   
    print('\nF1 score:')
    f1_score=sum(f1_score_dict)/len(f1_score_dict) if len(f1_score_dict) > 0 else 0
    print(f1_score)
    print('Precision:', precision)
    print('Recall:', recall)
    #print(f1_score_dict[:100])
    with open('output.txt', 'a', encoding='utf-8') as f:
        f.write(f'{precision}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the results folder')
    parser.add_argument('--hit', type=int, required=True, help='Path to the results folder')
    args = parser.parse_args()
    
    dataset_name="MedReason"
    #folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_deep_research'
    #folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_self_refine'
    folder_path = f'../results/{dataset_name}_deepseek-R1_10shot'
    #folder_path = f'../results/{dataset_name}_deepseek-R1_CoT_deep_research_cleaned'
    #folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_deep_research_cleaned2'
    folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_RAG10'
    folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_RAG_trace'
    #folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_all'
    #folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_all-search'
    folder_path = f'../results/{dataset_name}_deepseek-R1_10shot_SOAP1'
    folder_path = args.folder_path
    IND_list=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 154, 155, 156, 157, 158, 160, 162, 164, 165, 166, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 194, 195, 196, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 222, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 346, 347, 348, 349, 350, 351, 352, 354, 355, 357, 358, 359, 360, 361, 363, 364, 365, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 395, 397, 398, 399, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 413, 414, 415, 416, 417, 418, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 450, 451, 453, 454, 455, 456, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 476, 477, 478, 479, 480, 481, 482, 484, 486, 489, 490, 491, 492, 493, 494, 495, 496, 497, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 511, 512, 514, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 533, 534, 535, 536, 537, 538, 539, 540, 541, 544, 545, 546, 547, 548, 549, 550, 552, 553, 554, 555, 556, 557, 558, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 574, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 592, 593, 594, 596, 597, 598, 599, 600, 601, 602, 604, 605, 606, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 645, 646, 648, 649, 650, 651, 652, 653, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 673, 674, 675, 676, 677, 678, 679, 680, 681, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 711, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 730, 731, 732, 733, 734, 735, 736, 737, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 757, 758, 759, 760, 761, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 789, 790, 791, 792, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 810, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 842, 843, 844, 845, 847, 848, 849, 850, 852, 853, 854, 855, 856, 857, 859, 860, 861, 862, 863, 864, 865, 867, 868, 869, 870, 871, 872, 873, 874, 876, 877, 878, 879, 880, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896]

    print(folder_path)
    EM_dict, f1_score_dict, precision,recall = compute_metrics(folder_path,dataset_name,IND_list,hit=args.hit)
    print_results(EM_dict, f1_score_dict,precision,recall)
    

if __name__ == '__main__':
    main()