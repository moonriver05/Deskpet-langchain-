[README.md](https://github.com/user-attachments/files/28863794/README.md)
# 久远寺有珠桌宠 2.0 —— 带记忆与灵魂的桌面伙伴

<p align="center">
  <img src="./有珠.gif" width="180" alt="有珠桌宠动画">
  <img src="./猫猫有珠.gif" width="180" alt="猫猫有珠动画">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-6FA8DC?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyQt5" src="https://img.shields.io/badge/PyQt5-Desktop-8EA8C3?style=for-the-badge">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-Agent-3D2B56?style=for-the-badge">
  <img alt="Version" src="https://img.shields.io/badge/Version-2.0-1A1A1A?style=for-the-badge">
</p>

一个基于 PyQt5 的本地桌面宠物程序，拥有记忆系统、向量检索知识库、语音合成、情感表情包、待办清单、主动关怀和定时提醒等功能。

你可以把她当作一个一直挂在桌面上的陪伴式 Agent：她会记住你经常提到的事，理解你最近在忙什么，在合适的时候提醒你休息，也可以帮你整理待办、检索文档、读出回复。

仓库地址：[moonriver05/Deskpet-langchain-](https://github.com/moonriver05/Deskpet-langchain-)

## 目录

- [功能特点](#-功能特点)
- [2.0 更新亮点](#-20-更新亮点)
- [安装与依赖](#-安装与依赖)
- [首次使用配置](#-首次使用配置)
- [使用指南](#-使用指南)
- [表情包同步](#-表情包同步)
- [项目结构](#-项目结构)
- [第三方依赖与致谢](#-第三方依赖与致谢)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

## ✨ 功能特点

### 🧠 记忆系统

- **短期记忆**：保存近期对话中值得回忆的信息，并参与当前聊天检索。
- **长期记忆**：短期记忆达到一定重要度后迁移为长期记忆，作为后续用户画像和训练数据来源。
- **用户画像**：长期记忆不会直接堆进 prompt，而是被提炼成更紧凑的画像，例如学习习惯、身体情况、互动偏好。
- **混合检索**：结合 Chroma 向量召回、中文词面召回、最近上下文和本地融合打分，减少“明明不相关却被拉出来”的情况。
- **DeepSeek 重排**：可选用 DeepSeek 对候选记忆做小请求重排，失败时自动退回本地分数。

### 📚 知识库系统

- **多格式支持**：`.txt`、`.md`、`.pdf`、`.docx`。
- **Markdown 智能切分**：尽量保留标题和章节结构，自动拆分为带上下文的语义块。
- **相邻片段扩展**：命中文档片段后，会拉取附近内容，避免只给模型一小段断裂文本。

### 💬 智能对话

- **火山方舟 LLM 驱动**：兼容 OpenAI API 格式，可接入豆包等模型。
- **短期上下文**：保留最近对话，让有珠知道刚刚聊过什么。
- **现实边界**：角色设定和本地能力分开，尽量避免她说出“让小使魔碰你手背”这类现实中做不到的动作。
- **反馈按钮**：回复支持点赞/点踩，后续可以作为偏好模型或回复重排的数据。

### 🖼️ 情感表情包

- **情绪匹配**：根据回复情绪选择合适的表情包。
- **本地图集**：可以把表情包按情绪放在 `memes/` 目录下。
- **COS 图床**：配置腾讯云 COS 后，可以同步表情包并在聊天气泡里显示。

### 🔊 语音合成

- **GPT-SoVITS 接入**：回复可以自动合成语音。
- **一键播放**：聊天气泡旁有播放按钮，可播放或重试。
- **音色配置**：可以指定参考音频和权重，让声音更贴合角色。

### ✅ 待办清单

- **手动管理**：支持添加、编辑、删除、完成待办。
- **聊天写入**：当你明确说“帮我记一下”“提醒我”时，可以自动写入待办。
- **作业同步**：支持同步未完成作业并转成待办事项。
- **筛选搜索**：支持按状态、分类和关键词快速查找。

### ⏱️ 定时提醒

- **喝水提醒**：本地定时触发，不需要消耗大模型 token。
- **久坐提醒**：长时间运行时自动提醒你站起来动一动。
- **专注倒计时**：可以启动专注/倒计时窗口。
- **主动关怀**：和普通定时提醒分开，根据最近上下文生成更像陪伴的消息。

### 🎨 桌面宠物

- **GIF 动画支持**：默认读取 `有珠.gif` 作为桌宠动画。
- **右键菜单**：打开聊天、待办、记忆管理、设置等窗口。
- **拖拽移动**：按住左键即可拖动。
- **暗色魔女风 UI**：部分窗口已调整为黑、雾霾蓝、暗紫的极简暗色主题。

## 🌙 2.0 更新亮点

- 从单文件逐步拆成模块，当前已拆出 `pet_core/`、`pet_memory/`、`pet_services/`、`pet_features/`。
- 短期记忆、长期记忆、用户画像分层，不再把所有记忆粗暴塞进 prompt。
- 记忆检索加入最近上下文、中文词面召回、2/3-gram、MySQL LIKE、本地融合分数和 DeepSeek 重排。
- 画像精炼加入 evidence/claim 结构，减少重复画像和旧证据残留。
- Chroma 写入/删除失败时进入同步队列，后续自动修复。
- 增加本地能力注册表，让模型更清楚有珠能做什么、不能做什么。
- 待办系统独立成模块，后续更方便维护。
- 增加点赞/点踩反馈，为后续个性化模型训练做准备。

## 📦 安装与依赖

### 环境要求

- Windows
- Python 3.10+
- MySQL 5.7 / 8.0
- Docker
- Chroma MCP 容器

可选：

- GPT-SoVITS：用于语音合成
- 腾讯云 COS：用于表情包图床

### 安装 Python 依赖

```bash
pip install PyQt5 pymysql requests langchain-openai langchain-core openai zhdate jieba
pip install PyMuPDF python-docx qcloud-cos
```

### Docker 启动 Chroma

```bash
docker run -d --name chroma-mcp --restart unless-stopped ^
  -v chroma_pet_data:/chroma_data ^
  --entrypoint sleep mcp/chroma infinity
```

程序会复用这个容器：

```bash
docker exec -i chroma-mcp chroma-mcp --client-type persistent --data-dir /chroma_data
```

### GPT-SoVITS（可选）

将 GPT-SoVITS 工程放入 `voice/` 目录后，启动对应 API 服务，例如：

```bash
cd voice/GPT-SoVITS-v2pro-20250604
runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
```

## ⚙️ 首次使用配置

桌宠第一次启动时会弹出设置窗口。常用配置如下：

| 配置项 | 说明 |
| --- | --- |
| MySQL | 存储短期记忆、长期记忆、画像、待办等数据 |
| 火山方舟 API Key | 主聊天模型 |
| DeepSeek API Key | 可选，用于画像精炼和记忆重排 |
| Chroma 容器名 | 默认 `chroma-mcp` |
| 腾讯云 COS | 可选，用于表情包图床 |
| GPT-SoVITS | 可选，用于语音合成 |
| 优学院账号 | 可选，用于同步作业待办 |

配置保存后会写入本地 `pet_config.json`。

## 🚀 使用指南

### 启动桌宠

Windows 下可以双击：

```text
启动桌宠.bat
```

也可以手动运行：

```bash
python pet.py
```

### 交互方式

| 操作 | 功能 |
| --- | --- |
| 左键拖拽 | 移动桌宠 |
| 右键单击 | 打开功能菜单 |
| 聊天窗口 | 对话、发送图片、播放语音、点赞/点踩 |
| 待办窗口 | 管理待办、同步作业 |
| 记忆管理 | 查看和编辑短期记忆、长期记忆、用户画像 |
| 设置窗口 | 配置模型、数据库、TTS、COS 等 |

### 聊天功能

1. 右键打开聊天窗口。
2. 输入文字或发送图片。
3. 有珠会结合角色设定、当前输入、用户画像、相关短期记忆、知识库和最近上下文回复。
4. 如果配置了 TTS，可以点击气泡旁的播放按钮听语音。
5. 可以给回复点赞/点踩，作为后续个性化训练数据。

### 待办清单

- 手动添加、编辑、删除待办。
- 聊天时明确要求记录事项，可以自动写入待办。
- 支持作业同步。
- 支持搜索和筛选。

### 记忆管理

- 可以查看短期记忆、长期记忆、用户画像。
- 可以手动添加、编辑、删除记忆。
- 修改短期记忆时会同步到 Chroma。
- 修改长期记忆后会重新精炼画像，避免旧画像残留。

## 🖼️ 表情包同步

如果你配置了腾讯云 COS，可以把表情包按情绪放入 `memes/` 目录：

```text
memes/
├── happy/
├── sad/
├── angry/
├── tired/
├── surprised/
└── neutral/
```

然后在聊天窗口中点击同步按钮，把本地表情包同步到图床。有珠回复时会根据情绪选择合适的图片。

推荐按英文情绪名分类，方便模型和程序匹配。没有配置 COS 时，聊天功能仍然可用，只是不会显示远程表情包。

## 📁 项目结构

```text
.
├── pet.py                    # 主程序入口
├── 启动桌宠.bat              # Windows 双击启动
├── pet_core/                 # 配置、角色设定、上下文、主动关怀、计时解析
├── pet_memory/               # 记忆、画像、检索、Chroma 同步、数据库初始化
├── pet_services/             # Chroma、知识库、TTS 等服务封装
├── pet_features/             # 待办系统等功能模块
├── pet_config.json           # 本地配置文件
├── conversation_history.json # 最近对话上下文
├── todo_data.json            # 待办数据
├── feedback_data.jsonl       # 点赞/点踩反馈
├── memes/                    # 本地表情包目录
├── voice/                    # GPT-SoVITS 工程目录
├── tts_cache/                # 语音缓存
├── 有珠.png                  # 桌宠静态图
├── 有珠.gif                  # 桌宠动画
└── 猫猫有珠.gif              # 备用动画
```

### 数据存储说明

| 数据类型 | 存储位置 | 说明 |
| --- | --- | --- |
| 短期记忆 | MySQL + Chroma | 当前聊天检索使用 |
| 长期记忆 | MySQL | 用户画像和后续训练数据来源 |
| 用户画像 | MySQL | 进入 prompt 的长期个性化摘要 |
| 知识库 | Chroma | 上传文档切片 |
| 最近对话 | `conversation_history.json` | 近期上下文 |
| 待办清单 | `todo_data.json` | 本地待办 |
| 反馈数据 | `feedback_data.jsonl` | 点赞/点踩 |

## 🙏 第三方依赖与致谢

本项目在开发中参考了以下优秀项目：

- [astrbot_plugin_meme_manager](https://github.com/anka-afk/astrbot_plugin_meme_manager)：表情包管理思路。
- [astrbot_plugin_angel_memory](https://github.com/kawayiYokami/astrbot_plugin_angel_memory)：记忆系统设计启发。
- [Shinsekai](https://github.com/RachelForster/Shinsekai)：陪伴式 Agent 的交互体验参考。
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：少样本语音合成。

核心技术栈：

- PyQt5
- LangChain
- Chroma + MCP
- MySQL
- 火山方舟 / DeepSeek
- 腾讯云 COS
- GPT-SoVITS

## ❓ 常见问题

### 启动时数据库连接失败？

检查 MySQL 是否已启动，以及设置窗口里的 host、port、user、password 是否正确。

### 记忆检索为空？

第一次使用时记忆为空是正常的。确认 Chroma 容器正在运行，并且聊天中已经产生可存储的记忆。

### 回复没有表情包？

需要配置腾讯云 COS 并同步表情包。没有配置时仍然可以正常聊天。

### 回复没有声音？

需要单独启动 GPT-SoVITS API，并在设置窗口中填写 TTS 地址、参考音频和权重。

### 主动关怀为什么没有在电脑关机后继续？

这是本地桌宠程序，需要电脑和程序保持运行。没有运行时不会在后台执行主动关怀或定时提醒。

## 📄 许可证

本项目基于 **Apache License 2.0** 开源。

```text
Copyright 2026 pet_desktop

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

如果这个项目对你有帮助，欢迎给个 Star，也欢迎提交 Issue 记录 bug、建议和新的想法。
