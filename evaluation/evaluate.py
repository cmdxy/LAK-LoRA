# coding=utf-8

import json
import sys
import os

from evaluators import calc_info_extract_task_scores, calc_cls_task_scores, calc_nlg_task_scores, \
    calc_nlg_task_scores_by_sessions


error_msg={
    1: "There are missing predictions in the submission, please check again!",
    2: "Predictions are in the wrong format, please check again! ",
    3: "It seems there are missing predictions or the predicted samples are in the wrong order, please check again! ",
    4: "Error in calculating metrics!",

    10: "test_predictions.json file not in submission, please check again!",
    11: "results.json file not in submission, please check again!",
    12: "post_generate_process.py file not in submission, please check again!",
    13: "loading results.json file fails, please check again!",
    99: "Other error unknown!"
}


def dump_2_json(info, path):
    with open(path, 'a') as output_json_file:
        output_json_file.write(json.dumps(info, indent=2, ensure_ascii=False) + '\n')
        # json.dump(info, output_json_file, indent=2, ensure_ascii=False)


def report_error_msg(detail, show_msg, out_p):
    error_dict = dict()
    error_dict['errorDetail'] = detail
    error_dict['errorMsg'] = show_msg
    error_dict['score'] = 0
    error_dict['scoreJson'] = {}
    error_dict['success'] = False
    dump_2_json(error_dict, out_p)


def report_score(score_map, out_p, in_p):
    result = dict()
    result['success'] = True
    result['file'] = in_p
    result['score'] = score_map['score']
    result['scoreJson'] = score_map

    dump_2_json(result, out_p)


def calc_scores(dict_gt, dict_pred, out_path):

    scores = {
        "CMeEE-V2": {},
        "CMeIE-V2": {},
        "CHIP-CDN": {},
        "CHIP-CDEE": {},
        "IMCS-V2-NER": {},
        "CHIP-MDCFNPC": {},
        "IMCS-V2-SR": {},
        "IMCS-V2-DAC": {},
        "IMCS-V2-MRG": {},
        "CHIP-CTC": {},
        "CHIP-STS": {},
        "KUAKE-IR": {},
        "KUAKE-QIC": {},
        "KUAKE-QQR": {},
        "KUAKE-QTR": {},
        "MedDG": {}
    }

    success_flag = 1

    for task_name in scores.keys():

        assert task_name in dict_gt

        if task_name not in dict_pred:
            report_error_msg(error_msg[1], error_msg[1], out_path)
            success_flag = 0
            break

        gts = dict_gt[task_name]
        preds = dict_pred[task_name]
        if not len(gts) == len(preds):
            report_error_msg(error_msg[1], error_msg[1], out_path)
            success_flag = 0
            break

        for gt_inst, pred_inst in zip(gts, preds):
            if not isinstance(pred_inst, dict):
                report_error_msg(error_msg[2], error_msg[2], out_path)
                success_flag = 0
                break

            if "sample_id" not in pred_inst:
                report_error_msg(error_msg[2], error_msg[2], out_path)
                success_flag = 0
                break

            if gt_inst.get("sample_id") != pred_inst.get("sample_id"):
                report_error_msg(error_msg[3], error_msg[3], out_path)
                success_flag = 0
                break

        if task_name in ["CMeEE-V2", "CMeIE-V2", "CHIP-CDN", "CHIP-CDEE", "IMCS-V2-NER", "IMCS-V2-SR",
                         "CHIP-MDCFNPC",
                         ]:
            try:
                precision, recall, f1 = calc_info_extract_task_scores(
                    gts,
                    preds
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break


        # "CHIP-STS"
        elif task_name in ["CHIP-STS", ]:
            try:
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=["是的", "不是"],
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break


        elif task_name in ["CHIP-CTC", ]:
            try:
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=['非上述类型', '疾病', '症状(患者感受)',
                                 '体征(医生检测）', '怀孕相关', '肿瘤进展',
                                 '疾病分期', '过敏耐受', '器官组织状态',
                                 '预期寿命', '口腔相关', '药物',
                                 '治疗或手术', '设备', '护理',
                                 '诊断', '实验室检查', '风险评估',
                                 '受体状态', '年龄', '特殊病人特征',
                                 '读写能力', '性别', '教育情况',
                                 '居住情况', '种族', '知情同意',
                                 '参与其它试验', '研究者决定', '能力',
                                 '伦理审查', '依存性', '成瘾行为',
                                 '睡眠', '锻炼', '饮食', '酒精使用',
                                 '性取向', '吸烟状况', '献血',
                                 '病例来源', '残疾群体', '健康群体',
                                 '数据可及性', "含有多个类别"],
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in ["IMCS-V2-DAC", ]:
            try:
                list_labels = [
                    '非上述类型',
                    '关于症状的询问', '关于症状的回答',
                    '关于病因的询问', '关于病因的回答',
                    '关于个人基本信息的询问', '关于个人基本信息的回答',
                    '关于已有检查和治疗的提问', '关于已有检查和治疗的回答',
                    '关于用药建议的提问', '关于用药建议的解答',
                    '关于就医建议的提问', '关于就医建议的解答',
                    '关于注意事项的提问', '关于注意事项的解答',
                    '给出诊断',
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in ["KUAKE-IR", ]:
            try:
                list_labels = ["相关", "不相关"]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in ["KUAKE-QIC", ]:
            try:
                list_labels = [
                    '非上述类型',
                    "病情诊断",
                    "病因分析",
                    "治疗方案",
                    "就医建议",
                    "指标解读",
                    "疾病描述",
                    "后果表述",
                    "注意事项",
                    "功效作用",
                    "医疗费用",


                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=True,
                )
                scores[task_name] = {
                    "macro-precision": precision,
                    "macro-recall": recall,
                    "macro-f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in ["KUAKE-QTR", ]:
            try:
                list_labels = [
                    "完全不匹配或者没有参考价值",
                    "很少匹配有一些参考价值",
                    "部分匹配",
                    "完全匹配",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in ["KUAKE-QQR", ]:
            try:
                list_labels = [
                    "完全一致",
                    "后者是前者的语义子集",
                    "后者是前者的语义父集",
                    "语义无直接关联",
                ]
                precision, recall, f1 = calc_cls_task_scores(
                    gts,
                    preds,
                    list_labels=list_labels,
                    return_macro=False,
                )
                scores[task_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in [
            "MedDG",
        ]:
            try:
                rouge1, rouge2, rougeL = calc_nlg_task_scores(
                    gts,
                    preds,
                )
                scores[task_name] = {
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougeL,
                }
            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

        elif task_name in [
            "IMCS-V2-MRG",
        ]:
            try:
                rouge1, rouge2, rougeL = calc_nlg_task_scores_by_sessions(
                    gts,
                    preds,
                )
                scores[task_name] = {
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougeL,
                }
            except Exception as e:
                print(e)
                report_error_msg(error_msg[4], str(e), out_path)
                success_flag = 0
                break

    total_score = 0.0
    if success_flag:
        for task_name in scores.keys():
            if "rougeL" in scores[task_name]:
                total_score += scores[task_name].get("rougeL", 0.0)
            elif "macro-f1" in scores[task_name]:
                total_score += scores[task_name].get("macro-f1", 0.0)
            else:
                total_score += scores[task_name].get("f1", 0.0)

    scores["Overall"] = total_score / len(scores.keys())
    print("scores for all tasks: ", scores)

    score_map = {}
    for task_name in scores.keys():
        if task_name == "Overall":
            continue

        if task_name in ["CHIP-CTC", "KUAKE-QIC", "IMCS-V2-DAC"]:
            score_map[f"{task_name}-Macro-F1"] = scores[task_name].get("macro-f1", 0.0)
        elif task_name in ["MedDG", "IMCS-V2-MRG"]:
            score_map[f"{task_name}-RougeL"] = scores[task_name].get("rougeL", 0.0)
        else:
            score_map[f"{task_name}-Micro-F1"] = scores[task_name].get("f1", 0.0)
    score_map["score"] = scores["Overall"]

    print(score_map)
    return score_map, success_flag

if __name__ == "__main__":
    debug_mode = 0

    success_flag = 1

    # 加载金标准文件和预测文件
    submit_dir = ""
    standard_path = ""
    out_path = ""

    with open(out_path, 'w') as output_json_file:
        pass

    for file_name in os.listdir(submit_dir):
        print(file_name)
        if ".json" not in file_name:
            tmp = os.path.join(submit_dir, file_name)
            for sub_file_name in os.listdir(tmp):
                submit_path = os.path.join(tmp, sub_file_name)
                dict_pred = None
                try:
                    dict_pred = json.load(
                        open(submit_path, "r", encoding="utf-8")
                    )
                except Exception as e:
                    print(e)
                    success_flag = 0
                    check_code = 13
                    report_error_msg(error_msg[check_code], str(e), out_path)

                dict_gt = json.load(
                    open(standard_path, "r", encoding="utf-8")
                )

                if success_flag:
                    try:
                        score_map, success_flag = calc_scores(
                            dict_gt, dict_pred, out_path
                        )

                        if success_flag:
                            score_map = {key: value * 100 for key, value in score_map.items()}
                            report_score(score_map, out_path, submit_path)


                    except Exception as e:
                        print(e)
                        success_flag = 0
                        check_code = 99
                        report_error_msg(error_msg[check_code], str(e), out_path)
        else:
            submit_path = os.path.join(submit_dir, file_name)
            dict_pred = None
            try:
                dict_pred = json.load(
                    open(submit_path, "r", encoding="utf-8")
                )
            except Exception as e:
                print(e)
                success_flag = 0
                check_code = 13
                report_error_msg(error_msg[check_code], str(e), out_path)

            dict_gt = json.load(
                open(standard_path, "r", encoding="utf-8")
            )


            if success_flag:
                try:
                    score_map, success_flag = calc_scores(
                        dict_gt, dict_pred, out_path
                    )

                    if success_flag:
                        score_map = {key: value * 100 for key, value in score_map.items()}
                        report_score(score_map, out_path, submit_path)


                except Exception as e:
                    print(e)
                    success_flag = 0
                    check_code = 99
                    report_error_msg(error_msg[check_code], str(e), out_path)