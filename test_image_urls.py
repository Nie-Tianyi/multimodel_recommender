import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import random
import time

def test_specific_urls():
    """测试特定的问题URL和正常URL"""
    print("=== 测试有问题的URL ===")
    problematic_urls = [
        ("B00J878GES", "https://images-na.ssl-images-amazon.com/images/I/815UqGINqML.jpg"),
        ("B07QV9NCQZ", "https://images-na.ssl-images-amazon.com/images/I/C1GYjNofiGS._SL1600_.jpg")
    ]
    
    for asin, url in problematic_urls:
        print(f"\n测试 ASIN: {asin}")
        print(f"URL: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"内容大小: {len(response.content)} 字节")
                try:
                    img = Image.open(BytesIO(response.content))
                    print(f"图像信息: {img.format}, {img.size}, {img.mode}")
                except Exception as img_e:
                    print(f"图像解析错误: {img_e}")
            else:
                print(f"响应头: {dict(response.headers)}")
        except Exception as e:
            print(f"请求错误: {e}")
        time.sleep(1)
    
    print("\n=== 测试随机正常URL（验证可访问性）===")
    
    # 从有图像的商品中随机选择10个URL
    try:
        df = pd.read_csv('all_items_with_images.csv')
        if len(df) > 0:
            sample_size = min(10, len(df))
            sample = df.sample(sample_size)
            
            for idx, row in sample.iterrows():
                print(f"\n测试 ASIN: {row['asin']}")
                print(f"URL: {row['url']}")
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(row['url'], headers=headers, timeout=10)
                    print(f"状态码: {response.status_code}")
                    if response.status_code == 200:
                        print(f"内容大小: {len(response.content)} 字节")
                        try:
                            img = Image.open(BytesIO(response.content))
                            print(f"图像信息: {img.format}, {img.size}, {img.mode}")
                        except Exception as img_e:
                            print(f"图像解析错误: {img_e}")
                except Exception as e:
                    print(f"请求错误: {e}")
                time.sleep(1)
    except Exception as e:
        print(f"读取CSV文件错误: {e}")
    
    print("\n=== 图像覆盖率统计 ===")
    try:
        df_all = pd.read_csv('item_metadata.csv')
        df_with_images = pd.read_csv('all_items_with_images.csv')
        
        total_items = len(df_all)
        items_with_images = len(df_with_images)
        
        print(f"总商品数: {total_items}")
        print(f"有图像商品数: {items_with_images}")
        print(f"图像覆盖率: {items_with_images/total_items*100:.1f}%")
        print(f"无图像商品数: {total_items - items_with_images}")
    except Exception as e:
        print(f"统计错误: {e}")

def generate_test_urls_file():
    """生成测试URL文件供用户手动测试"""
    try:
        df = pd.read_csv('all_items_with_images.csv')
        
        # 选择不同类型的URL进行测试
        test_urls = []
        
        # 添加问题URL
        test_urls.append({
            'asin': 'B00J878GES',
            'url': 'https://images-na.ssl-images-amazon.com/images/I/815UqGINqML.jpg',
            'type': 'problematic',
            'expected_status': '404'
        })
        
        test_urls.append({
            'asin': 'B07QV9NCQZ',
            'url': 'https://images-na.ssl-images-amazon.com/images/I/C1GYjNofiGS._SL1600_.jpg',
            'type': 'problematic',
            'expected_status': '404'
        })
        
        # 添加随机正常URL
        normal_samples = df.sample(min(20, len(df)))
        for idx, row in normal_samples.iterrows():
            test_urls.append({
                'asin': row['asin'],
                'url': row['url'],
                'type': 'normal',
                'expected_status': '200'
            })
        
        # 保存到CSV
        test_df = pd.DataFrame(test_urls)
        test_df.to_csv('manual_test_urls.csv', index=False)
        print(f"已生成 {len(test_urls)} 个测试URL到 manual_test_urls.csv")
        
        # 生成HTML文件便于点击测试
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>图像URL测试页面</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .url-container { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .problematic { background-color: #ffe6e6; }
        .normal { background-color: #e6ffe6; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .status { font-weight: bold; }
        .status-404 { color: #cc0000; }
        .status-200 { color: #00cc00; }
    </style>
</head>
<body>
    <h1>图像URL测试页面</h1>
    <p>点击以下链接测试URL可访问性：</p>
"""
        
        for item in test_urls:
            status_class = f"status-{item['expected_status']}"
            container_class = item['type']
            
            html_content += f"""
    <div class="url-container {container_class}">
        <p><strong>ASIN:</strong> {item['asin']}</p>
        <p><strong>类型:</strong> {item['type']}</p>
        <p><strong>预期状态:</strong> <span class="status {status_class}">{item['expected_status']}</span></p>
        <p><a href="{item['url']}" target="_blank">{item['url']}</a></p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open('test_urls.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"已生成HTML测试页面 test_urls.html")
        
    except Exception as e:
        print(f"生成测试文件错误: {e}")

if __name__ == '__main__':
    print("图像URL测试工具")
    print("=" * 50)
    
    test_specific_urls()
    print("\n" + "=" * 50)
    
    print("\n生成测试文件...")
    generate_test_urls_file()
    
    print("\n" + "=" * 50)
    print("使用说明:")
    print("1. 直接运行此脚本测试特定URL")
    print("2. 查看 manual_test_urls.csv 获取测试URL列表")
    print("3. 打开 test_urls.html 在浏览器中点击测试")
    print("4. 有问题的URL已用红色高亮显示")