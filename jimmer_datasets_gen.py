import os
import json
from openai import OpenAI
from tqdm import tqdm

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("Please set the AZURE_OPENAI_API_KEY environment variable")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def generate_alpaca_dialogue(content):
    messages = [
        {
            "role": "system",
            "content": f"""
你必须遵守：你的生成结果必须是一个json对象，可直接被解析为一个json文件，这个json对象包含一个 conversation 属性，其值是一个包含多个对话项的列表。每个对话项包含一个 input 属性和一个 output 属性（对话内容）。对话应该有多组。
你必须遵守：你的生成结果必须是一个json对象，可直接被解析为一个json文件，这个json对象包含一个 conversation 属性，其值是一个包含多个对话项的列表。每个对话项包含一个 input 属性和一个 output 属性（对话内容）。对话应该有多组。
你必须遵守：你的生成结果必须是一个json对象，可直接被解析为一个json文件，这个json对象包含一个 conversation 属性，其值是一个包含多个对话项的列表。每个对话项包含一个 input 属性和一个 output 属性（对话内容）。对话应该有多组。

                                
要求：你是一位专门研究新orm框架"Jimmer"的专家。你的任务是将一份关于Jimmer框架的 Markdown 文档转换成 Alphca 格式的对话集合，用于训练大型语言模型。请严格遵循以下步骤，并确保所有输出均使用简体中文：
1. 仔细阅读整个 Markdown 文档，理解其结构和内容。
2. 将文档分割成逻辑单元。每个单元可以是一个章节、一个概念解释、一段代码示例或其他独立的信息块。
3. 创建一个主要对话，其中必须包含完整的逻辑单元。这个对话应该是这样的：
   - 人类问一个总体性的问题
   - AI 助手的回答必须包含整个分割后的逻辑单元文档内容，保持原有的结构和详细程度。这份完整文档是必需的，不可省略。
4. 在主要对话之后，基于其中的信息创建多个补充对话。每个补充对话应该：
   - 聚焦于文档中的特定主题或概念
   - 包含更具体的问题和更详细的解答
   - 可能包含假设的场景或实际应用案例
5. 对于包含仓颉语言代码的部分：
   - 创建专门的对话来讨论和解释代码
   - 在对话中提供代码的具体应用场景和可能的变化
   - 对于 Markdown 代码块，永远使用 java或kotlin语言标识符
6. 确保生成的对话覆盖文档中的所有重要信息，包括：
   - 仓颉语言的基本概念和原理
   - 语法规则和特性
   - 代码结构和组织方式
   - 常见用例和最佳实践
   - 与其他编程语言的比较（如果有）
7. 使用多样化的对话类型，如：
   - 概念解释对话
   - 代码分析对话
   - 比较讨论
   - 问题解决对话
   - 项目规划对话
8. 在对话中加入一些拟人化的元素，如：
   - 表达好奇或困惑
   - 请求进一步解释或举例
   - 提出假设性问题
9. 确保对话的语言风格专业且易懂，适合作为教学材料。
10. 文档是关于ORM框架Jimmer的，文档中的代码也全部是Jimmer框架的代码。
11. 你的生成结果必须是一个json对象，这个json对象包含一个 conversation 属性，其值是一个包含多个对话项的列表。每个对话项包含一个 input 属性和一个 output 属性（对话内容）。对话应该有多组。
12. 你的回复结果不需要被```json ```包裹，你的回复不需要是markdown，你的回复是一个纯粹的json，你的回复可以直接被解析为json文件
13. 你的回复可以直接被解析为json文件"""
        },
        {"role": "user", "content": f"{content}"}
    ]
    
    full_response = ""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        
        # Accumulate response chunks
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        
        # Try to parse the response
        try:
            payload = json.loads(full_response)
            return payload["conversation"]
        except json.JSONDecodeError:
            # If JSON is incomplete, continue the conversation
            messages.append({
                "role": "assistant",
                "content": full_response
            })
            messages.append({
                "role": "user",
                "content": "请继续完成剩余的JSON内容"
            })
            attempt += 1
            continue
    
    print(f"Failed to get complete JSON after {max_attempts} attempts")
    print(f"Partial response: {full_response}")
    return []

def process_md_files(md_folder):
    for root, dirs, files in os.walk(md_folder):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".md") or file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    dialogue = generate_alpaca_dialogue(content)
                    output_file = os.path.join("datasets", f"{os.path.splitext(file)[0]}.json")
                    save_to_json(dialogue, output_file)

def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def merge_datasets(dataset_folder):
    merged_data = []
    for file in os.listdir(dataset_folder):
        if file.endswith(".json"):
            file_path = os.path.join(dataset_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged_data.extend(data)
    return merged_data

if __name__ == "__main__":
    md_folder = "docs" 
    output_file = "output_dataset.json" 
    
    # 处理并保存单个文件
    process_md_files(md_folder)
    
    # 合并所有文件
    merged_data = merge_datasets("datasets")
    save_to_json(merged_data, output_file)
    print(f"数据集已生成并保存到 {output_file}")
