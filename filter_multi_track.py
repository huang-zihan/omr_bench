import csv

input_filename = 'music_scores_with_track_count.csv'
output_filename = 'filtered_music_scores.csv'

with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
     open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    writer.writeheader()  # 写入列标题
    
    for row in reader:
        try:
            # 将track_cnt转换为整数并比较
            if int(row['track_cnt']) >= 3:
                writer.writerow(row)
        except (ValueError, KeyError):
            # 处理无效数据或列名错误
            print(f"Skipping invalid row: {row}")
            continue

print(f"过滤完成！结果已保存至: {output_filename}")