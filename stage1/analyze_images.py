import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import ast
import random
import time
from tqdm import tqdm
import json

def extract_image_urls_from_images_field(images_str):
    """从images字段提取图像URL列表"""
    if pd.isna(images_str) or images_str == '[]' or images_str == '':
        return []
    
    try:
        # 尝试解析字符串为列表
        if isinstance(images_str, str):
            images_list = ast.literal_eval(images_str)
        else:
            images_list = images_str
            
        if not isinstance(images_list, list):
            return []
            
        urls = []
        for img_dict in images_list:
            if isinstance(img_dict, dict) and 'large_image_url' in img_dict:
                urls.append(img_dict['large_image_url'])
        return urls
    except Exception as e:
        print(f"解析images字段错误: {e}")
        return []

def test_url_accessibility(url, timeout=5):
    """测试URL可访问性"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if response.status_code == 200:
            # 尝试读取少量数据检查是否为有效图像
            content = response.content[:1024]  # 读取前1KB
            if len(content) > 0:
                # 尝试检查是否为图像
                try:
                    Image.open(BytesIO(content))
                    return True, response.status_code, "OK"
                except:
                    return True, response.status_code, "Not a valid image"
            return True, response.status_code, "Empty content"
        else:
            return False, response.status_code, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, None, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection error"
    except requests.exceptions.TooManyRedirects:
        return False, None, "Too many redirects"
    except Exception as e:
        return False, None, str(e)

def main():
    print("加载商品元数据...")
    df = pd.read_csv('item_metadata.csv')
    print(f"总商品数量: {len(df)}")
    
    # 提取图像URL
    print("提取图像URL...")
    image_urls_info = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        asin = row['asin']
        images_str = row['images']
        
        urls = extract_image_urls_from_images_field(images_str)
        if urls:
            # 随机选择一个图像URL（与原逻辑一致）
            selected_url = random.choice(urls)
            image_urls_info.append({
                'asin': asin,
                'url': selected_url,
                'total_images': len(urls)
            })
    
    print(f"有图像的商品数量: {len(image_urls_info)}")
    print(f"无图像的商品数量: {len(df) - len(image_urls_info)}")
    
    # 保存所有有图像的商品信息
    urls_df = pd.DataFrame(image_urls_info)
    urls_df.to_csv('all_items_with_images.csv', index=False)
    print(f"已保存有图像的商品列表到 all_items_with_images.csv")
    
    # 抽样测试URL可访问性
    print("\n抽样测试URL可访问性（测试前100个）...")
    sample_size = min(100, len(image_urls_info))
    sample_indices = random.sample(range(len(image_urls_info)), sample_size)
    
    test_results = []
    for i in tqdm(sample_indices, total=sample_size):
        item = image_urls_info[i]
        url = item['url']
        
        accessible, status_code, message = test_url_accessibility(url)
        
        test_results.append({
            'asin': item['asin'],
            'url': url,
            'accessible': accessible,
            'status_code': status_code,
            'message': message
        })
        
        # 避免请求过快
        time.sleep(0.1)
    
    # 分析结果
    results_df = pd.DataFrame(test_results)
    accessible_count = results_df['accessible'].sum()
    inaccessible_count = len(results_df) - accessible_count
    
    print(f"\n测试结果统计:")
    print(f"测试URL总数: {len(results_df)}")
    print(f"可访问URL数: {accessible_count} ({accessible_count/len(results_df)*100:.1f}%)")
    print(f"不可访问URL数: {inaccessible_count} ({inaccessible_count/len(results_df)*100:.1f}%)")
    
    # 按错误类型分组
    if inaccessible_count > 0:
        print("\n不可访问URL的错误类型:")
        error_counts = results_df[~results_df['accessible']]['message'].value_counts()
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
    
    # 保存测试结果
    results_df.to_csv('url_accessibility_test.csv', index=False)
    print(f"\n已保存测试结果到 url_accessibility_test.csv")
    
    # 提取有问题的URL
    problematic_urls = results_df[~results_df['accessible']]
    if len(problematic_urls) > 0:
        problematic_urls.to_csv('problematic_urls.csv', index=False)
        print(f"已保存 {len(problematic_urls)} 个有问题URL到 problematic_urls.csv")
        
        # 打印前10个有问题的URL供用户测试
        print("\n前10个有问题的URL（您可以尝试访问）:")
        for idx, row in problematic_urls.head(10).iterrows():
            print(f"ASIN: {row['asin']}")
            print(f"URL: {row['url']}")
            print(f"错误: {row['message']}")
            print("-" * 50)
    
    # 生成摘要报告
    report = {
        'total_items': len(df),
        'items_with_images': len(image_urls_info),
        'items_without_images': len(df) - len(image_urls_info),
        'images_coverage': len(image_urls_info) / len(df) * 100,
        'tested_urls': len(results_df),
        'accessible_urls': int(accessible_count),
        'inaccessible_urls': int(inaccessible_count),
        'accessibility_rate': accessible_count / len(results_df) * 100 if len(results_df) > 0 else 0
    }
    
    with open('image_coverage_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n已保存图像覆盖率报告到 image_coverage_report.json")
    print("\n报告摘要:")
    print(f"总商品数: {report['total_items']}")
    print(f"有图像商品数: {report['items_with_images']}")
    print(f"图像覆盖率: {report['images_coverage']:.1f}%")
    print(f"URL可访问率: {report['accessibility_rate']:.1f}%")

if __name__ == '__main__':
    main()