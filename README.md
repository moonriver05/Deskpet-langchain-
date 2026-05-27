# 久远寺有珠桌宠 —— 带记忆与灵魂的桌面伙伴

一个基于 PyQt5 的桌面宠物程序，拥有长期记忆系统、向量检索知识库、语音合成、待办清单等丰富功能。与《魔法使之夜》的久远寺有珠对话，她会记住你的喜好，随时间成长。


## 目录

- [功能特点](#-功能特点)
- [安装与依赖](#-安装与依赖)
- [首次使用配置](#-首次使用配置)
- [配置说明](#-配置说明)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [第三方依赖与致谢](#-第三方依赖与致谢)
- [许可证](#-许可证)


## ✨ 功能特点

### 🧠 长期记忆系统

- **语义检索优先**：基于 Chroma 向量数据库，确保召回的记忆与当前对话真正相关
- **艾宾浩斯遗忘机制**：记忆会随时间衰减，重要记忆被强化，过时记忆被遗忘
- **灵魂状态系统**：4 维能量槽动态调整 AI 行为——回忆深度、印象深度、表达欲望、创造力
- **记忆去重**：向量相似度 + 字面双重去重，避免重复存储

### 📚 知识库系统

- **多格式支持**：`.txt`、`.md`、`.pdf`、`.docx`
- **Markdown 智能切分**：保留章节结构，自动拆分为带上下文的语义块
- **Sentence-Window 扩展**：命中某个 chunk 后自动拉取相邻 chunk，保持上下文连贯

### 💬 智能对话

- **火山方舟 LLM 驱动**：兼容 OpenAI API 格式，可选 `doubao-1-5-pro` / `doubao-seed-2-0-mini` 等模型
- **短期对话记忆**：保留最近 10 轮对话上下文，避免反复问同一件事
- **情感表情包**：AI 自动判断情感并回复对应的表情包图片（需配置腾讯云 COS 图床）

### 🔊 语音合成

- **GPT-SoVITS 接入**：回复自动合成语音，气泡旁带 🔊 按钮可播放/重试
- **多音色支持**：可配置参考音频和微调权重，切换不同说话人音色

### ✅ 待办清单

- **AI 自动写入**：聊天时大模型智能判断是否需要记待办，自动写入清单
- **卡片化 UI**：优先级颜色标识、分类徽标、标签芯片、截止时间
- **优学院作业同步**：一键拉取未完成的在线作业，自动转为待办事项
- **A2UI 协议支持**：遵循 v0.9 规范，支持通过 UI 消息更新清单

### 🎨 桌面宠物

- **GIF 动画支持**：读取本地 `有珠.gif` 作为宠物动画
- **节日问候**：自动识别公历/农历节日并送上祝福
- **定时提醒**：久坐提醒、喝水提醒、随机闲聊
- **拖拽移动**：按住左键即可拖动


## 📦 安装与依赖

### 环境要求

- Python 3.10+
- MySQL 5.7 / 8.0（用于长期记忆存储）
- Docker（用于 Chroma 向量数据库）

### 安装 Python 依赖

```bash
pip install PyQt5 pymysql langchain-openai zhdate requests python-dotenv
pip install PyMuPDF python-docx jieba      # 可选，知识库文件解析
pip install qcloud-cos                     # 可选，图床支持
# 语音合成相关：可跳过，不影响基础聊天功能
```

### Docker 启动 Chroma

```bash
docker run -d --name chroma-mcp --restart unless-stopped \
  -v pet_desktop_chroma:/chroma_data --entrypoint sleep mcp/chroma infinity
```

### GPT-SoVITS（语音合成，可选）

1. 将 `GPT-SoVITS-v2pro-20250604` 放入 `voice/` 目录
2. 启动语音 API 服务：

```bash
cd voice/GPT-SoVITS-v2pro-20250604
runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
```


## ⚙️ 首次使用配置

桌宠启动时，会自动弹出**设置窗口**，你需要填写以下必填项：

| 配置项           | 获取方式                                                     |
| ---------------- | ------------------------------------------------------------ |
| MySQL 密码       | 你安装 MySQL 时设置的密码                                    |
| 火山方舟 API Key | 访问 [火山方舟控制台](https://www.volcengine.com/product/ark) 申请，格式 `ark-xxxxxx` |

配置保存后会自动写入 `pet_config.json`。

### 可选配置

| 模块               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| **腾讯云 COS**     | 表情包图床。需填写 SecretId、SecretKey、Bucket、地域、公网域名。留空则回复不带表情包 |
| **GPT-SoVITS**     | 语音合成。需指定参考音频路径、GPT/SoVITS 权重路径            |
| **优学院作业拉取** | 自动同步未完成作业到待办清单。需填写账号、密码及相应 API 地址 |
| **Chroma 容器名**  | 默认 `chroma-mcp`，一般无需修改                              |


## ⚙️ 配置说明

所有敏感配置（API Key、密码等）均存储在 `pet_config.json` 中，不会出现在代码里。

### 配置结构示例

```json
{
  "mysql": {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "pet_memory_db"
  },
  "ark": {
    "api_key": "ark-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxx",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model_main": "doubao-1-5-pro-32k-250115",
    "model_extractor": "doubao-seed-2-0-mini-260428",
    "model_tool": "doubao-seed-2-0-mini-260428"
  }
}
```

### 环境变量（可选）

支持以下环境变量覆盖配置：

- `CHROMA_MCP_CONTAINER`：Chroma 容器名
- `PET_TTS_API`：GPT-SoVITS API 地址
- 其他配置建议通过设置窗口填写，会自动写入 `pet_config.json`


## 🚀 使用指南

### 启动桌宠

```bash
python pet.py
```

### 交互方式

| 操作         | 功能                                                 |
| ------------ | ---------------------------------------------------- |
| **左键拖拽** | 移动桌宠                                             |
| **右键单击** | 打开功能菜单：聊天、待办清单、添加知识库、设置、退出 |
| **双击左键** | 快捷打开待办清单                                     |

### 聊天功能

1. 右键 → **💬 聊天** 打开聊天窗口
2. 输入文字或拖拽文件/图片发送
3. AI 会自动：
   - 提取关键词检索记忆
   - 判断是否需要存入新记忆
   - 识别情感并回复表情包（如已配置图床）
4. 语音合成：AI 回复后气泡旁会出现 🔊 按钮，点击播放语音

### 待办清单

- **手动添加**：在待办窗口底部输入框填写，选择优先级
- **AI 自动写入**：聊天时说“记得提醒我...”，AI 会自动写入清单
- **作业同步**：点击顶部 **🔄 获取作业** 自动拉取优学院未完成作业
- **筛选**：支持按状态（全部/未完成/已完成/今日）、分类、关键词搜索

### 知识库

- 右键 → **📚 添加知识库**，选择文档（.pdf/.docx/.txt/.md）
- 文档会自动切分并向量化，后续对话中可以检索相关段落

### 表情包同步（可选）

- 在 `memes/` 目录下按情感英文名创建子文件夹，放入图片
- 聊天窗口点击 **☁️** 按钮同步到腾讯云 COS 图床
- 之后 AI 回复时会自动匹配情感并显示对应表情包


## 📁 项目结构

```
.
├── pet.py                    # 主程序入口
├── pet_config.json           # 配置文件（自动生成，不要提交）
├── conversation_history.json # 短期对话历史
├── todo_data.json           # 待办清单数据
├── tts_cache/               # 语音缓存（退出时自动清理）
├── memes/                   # 本地表情包目录（按情感分类）
│   ├── happy/
│   ├── sad/
│   └── ...
├── voice/                   # GPT-SoVITS 工程目录
│   └── GPT-SoVITS-v2pro-20250604/
└── 有珠.gif                 # 宠物动画（可替换）
```

### 数据存储说明

| 数据类型 | 存储位置                    | 说明                     |
| -------- | --------------------------- | ------------------------ |
| 长期记忆 | MySQL + Chroma              | 用户画像、偏好、重要事实 |
| 短期对话 | `conversation_history.json` | 最近 10 轮对话           |
| 知识库   | Chroma（KB collection）     | 上传的文档切片           |
| 待办清单 | `todo_data.json`            | 未完成/已完成待办        |


## 🙏 第三方依赖与致谢

本项目在开发中参考了以下优秀项目：

- **[astrbot_plugin_meme_manager](https://github.com/anka-afk/astrbot_plugin_meme_manager)** —— AstrBot 表情包管理插件，参考了其功能组织与 README 风格
- **[astrbot_plugin_angel_memory](https://github.com/kawayiYokami/astrbot_plugin_angel_memory)** —— 记忆系统设计思路，特别是灵魂状态能量槽与三层认知架构
- **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** —— 强大的少样本语音合成引擎，用于实现桌宠语音回复

核心技术栈：

- **PyQt5** —— 跨平台 GUI 框架
- **LangChain** —— LLM 调用封装
- **Chroma** + **MCP** —— 向量数据库与检索
- **火山方舟** —— LLM 推理服务（OpenAI 兼容）
- **腾讯云 COS** —— 表情包图床
- **pymysql** —— MySQL 数据库连接


## 📄 许可证

本项目基于 **Apache License 2.0** 开源。

```
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


## ⚠️ 注意事项

1. **第一次启动会弹出设置窗口**，请务必填写 MySQL 密码和火山方舟 API Key
2. `pet_config.json` 包含 API 密钥和数据库密码，**请勿提交到 GitHub**
3. 建议将 `pet_config.json` 加入 `.gitignore`
4. Chroma 容器需提前运行，否则记忆检索不可用
5. GPT-SoVITS 语音合成需单独启动 API 服务，不影响基础聊天
6. 腾讯云 COS 配置缺失时，AI 回复仍正常，只是不带表情包图片


## ❓ 常见问题

**Q：启动时提示数据库连接失败？**  
A：检查 MySQL 是否已启动，以及在设置窗口中填写的密码是否正确。

**Q：AI 回复很慢或没反应？**  
A：检查火山方舟 API Key 是否正确填写，以及网络是否可达。

**Q：记忆好像不太相关？**  
A：长期记忆采用语义相似度主导的检索，第一次对话时记忆库为空是正常的。随着对话增加，记忆会逐渐积累。

**Q：想改宠物动画？**  
A：直接替换同目录下的 `有珠.gif` 即可，程序会自动识别并播放。

**Q：本地运行正常但打包成 exe 后报错？**  
A：检查 PyQt5 的 `Qt5/plugins` 路径是否正确；确保 `pet_config.json` 在可执行文件同目录下。


## 🗣️ 问题反馈

欢迎提交 [Issues](https://github.com/yourusername/pet_desktop/issues) 或 Pull Request。

---

**开发不易，如果这个项目对你有帮助，欢迎给个 Star ⭐**
