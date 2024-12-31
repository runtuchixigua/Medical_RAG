import json
import math

input_file = "medical.json"  # 输入的 JSON 文件路径
output_file = "output.txt"  # 输出的 txt 文件路径
count_num = 0
count = 0

# 打开并读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    # 打开输出文件进行写入
    with open(output_file, "w", encoding="utf-8") as out_f:
        # 逐行读取文件中的每个 JSON 对象
        for line in f:
            try:
                # 解析每一行 JSON 数据
                item = json.loads(line)

                # 提取信息
                name = item.get("name", "未知名称")
                desc = item.get("desc", "暂无描述")
                category = ", ".join(item.get("category", []))
                prevent = item.get("prevent", "暂无预防措施")
                cause = item.get("cause", "暂无病因信息")
                symptom = ", ".join(item.get("symptom", []))
                yibao_status = item.get("yibao_status", "未知")
                get_prob = item.get("get_prob", "未知")
                get_way = item.get("get_way", "未知")
                accompany = ", ".join(item.get("acompany", []))
                cure_department = ", ".join(item.get("cure_department", []))
                cure_way = ", ".join(item.get("cure_way", []))
                cure_lasttime = item.get("cure_lasttime", "未知")
                cured_prob = item.get("cured_prob", "未知")
                cost_money = item.get("cost_money", "未知")
                check = ", ".join(item.get("check", []))
                recommand_drug = ", ".join(item.get("recommand_drug", []))
                drug_detail = ", ".join(item.get("drug_detail", []))

                # 格式化文本内容
                txt_content = f"""
名称: {name}
描述: {desc}
类别: {category}
预防措施: {prevent}
病因: {cause}
症状: {symptom}\n
医保状态: {yibao_status}
获取概率: {get_prob}
获取途径: {get_way}
并发症: {accompany}
治疗科室: {cure_department}
治疗方式: {cure_way}
治疗周期: {cure_lasttime}
治愈概率: {cured_prob}
费用: {cost_money}
检查项目: {check}
推荐药物: {recommand_drug}
药物详情: {drug_detail}
                """

                # 写入文件，每个条目之间换行并分隔
                out_f.write(txt_content.strip())  # 去掉多余的空白
                count_num += len(txt_content)
                count += 1

                out_f.write("\n" + "\n")  # 分隔符

            except json.JSONDecodeError as e:
                print(f"跳过无效的 JSON 数据: {line.strip()}")

    print(f"所有内容已写入文件: {output_file}")
    print(count_num / count)
