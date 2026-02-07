# Seedream API Refiner Extension

Seedream API Refiner 是一个 Stable Diffusion webUI 扩展，利用 Seedream API 对生成的图像进行智能优化，提升图像质量和细节表现。

## 主要功能

- **智能图像优化**：使用 Seedream API 对 Stable Diffusion 生成的图像进行高质量优化
- **提示词翻译与优化**：内置 LLM 翻译功能，自动将中文提示词翻译并优化为英文，提升优化效果
- **参考图像支持**：支持上传参考图像，引导优化过程，保持风格一致性
- **高级图像预处理**：内置多种图像预处理工具（Canny、Depth、OpenPose、Lineart 等），可对参考图像进行处理
- **灵活的控制单元**：提供 4 个独立的控制单元，类似于 ControlNet 的使用方式
- **自动输出管理**：优化后的图像会自动保存到按日期组织的专用目录

## 技术架构

- **核心功能**：通过 OpenAI 兼容的 API 接口调用 Seedream API
- **用户界面**：使用 Gradio 构建直观的操作界面
- **图像处理**：集成多种预处理器，支持边缘检测、深度估计等操作
- **扩展集成**：无缝集成到 Stable Diffusion webUI 的扩展系统中

## 安装方法

1. **克隆仓库**：将本仓库克隆到你的 extensions 目录
   ```bash
   git clone https://github.com/yourusername/sd-SeedreamAPI.git extensions/sd-SeedreamAPI
   ```

2. **重启 webUI**：重启 Stable Diffusion webUI 以加载扩展

3. **安装依赖**：扩展会自动检查并安装所需的依赖（如 OpenAI 库）

## 使用指南

### 基本使用

1. **启用扩展**：在 webUI 界面中找到 "Seedream API Refiner" 部分，勾选 "Enable Seedream API Refiner" 复选框

2. **配置 API Key**：在 webUI 设置中找到 "Seedream API Refiner" 部分，输入你的 Seedream API Key

3. **设置提示词**：在 "Seedream Prompt" 文本框中输入优化提示词
   - 若不输入，默认使用 webUI 的正面提示词
   - 若 webUI 正面提示词也为空，会使用默认提示词："Optimize this image, add more details and hires"

4. **翻译与优化提示词**：点击 "Translate&Optimize" 按钮，使用 LLM 自动翻译并优化提示词

5. **生成图像**：像往常一样使用 Stable Diffusion 生成图像

6. **查看结果**：优化后的图像会自动保存到 `outputs-SeedreamRefiner/yy-mm-dd` 目录

### 高级功能：参考图像

1. **展开上传区域**：点击 "Upload Images" 展开参考图像设置

2. **启用参考图像**：勾选 "Enable Reference Images" 复选框

3. **上传参考图像**：点击 "Reference Images" 上传按钮，最多可上传 5 张参考图像

### 高级功能：控制单元

1. **展开图像预处理**：在 "Upload Images" 下点击 "Image Preprocessing" 展开控制单元设置

2. **配置控制单元**：每个控制单元包含以下设置：
   - **Enable**：启用该控制单元
   - **Image**：上传参考图像
   - **Preprocessor Category**：选择预处理器分类（全部、硬边缘、深度、姿态、线条）
   - **Preprocessor**：选择具体的预处理器（none、canny、depth_midas、openpose、lineart、lineart_anime）
   - **Enable Preview**：启用预处理结果预览
   - **Preview**：点击按钮查看预处理效果

3. **使用控制单元**：启用的控制单元会将预处理后的图像作为参考图像传递给 Seedream API

## 配置选项

### 主要设置

- **Enable Seedream API Refiner**：启用或禁用扩展
- **Seedream Prompt**：Seedream API 优化图像的提示词
- **Translate&Optimize**：翻译并优化提示词按钮

### 参考图像设置

- **Enable Reference Images**：启用或禁用参考图像
- **Reference Images**：上传参考图像（最多 5 张）

### 控制单元设置

- **Enable**：启用或禁用控制单元
- **Image**：上传控制图像
- **Preprocessor Category**：预处理器分类
- **Preprocessor**：具体预处理器
- **Enable Preview**：启用预览
- **Preview**：预览按钮

### 全局设置（webUI 设置页面）

- **Seedream API Key**：你的 Seedream API Key（必需）
- **Output Directory**：优化图像的输出目录

## 预处理器说明

| 预处理器 | 描述 | 适用场景 |
|---------|------|----------|
| none | 无预处理，直接使用原始图像 | 保持原始风格 |
| canny | 边缘检测，提取图像边缘 | 强化轮廓和结构 |
| depth_midas | 深度估计，生成深度图 | 增强空间感和立体感 |
| openpose | 姿态检测，提取人体姿态 | 保持人物姿态一致性 |
| lineart | 线条提取，生成线稿 | 强化线条艺术效果 |
| lineart_anime | 动漫风格线条提取 | 适用于动漫风格图像 |

## 注意事项

- **API Key 必需**：使用本扩展需要有效的 Seedream API Key
- **处理时间**：优化过程可能需要一定时间，具体取决于 API 响应速度
- **参考图像**：最多支持 4 张参考图像（包括控制单元处理的图像）
- **输出路径**：优化后的图像不会显示在 webUI 主界面，需到指定输出目录查看
- **预处理**：预处理器运算会在 CPU 上执行，以避免占用 GPU 资源
- **提示词优化**：使用 "Translate&Optimize" 功能可以获得更好的优化效果

## 故障排除

- **API Key 错误**：请确保输入了有效的 Seedream API Key
- **无输出**：检查输出目录是否存在且有写入权限
- **处理缓慢**：这可能是由于 API 响应时间较长，请耐心等待
- **预处理失败**：检查图像格式是否正确，确保图像不为空
- **参考图像无效**：确保上传的参考图像清晰可见，且与优化目标相关


## 许可证

MIT License

## 更新日志

- **v1.0.0**：初始版本，实现基本功能
- **v1.1.0**：添加控制单元功能，支持多种预处理器
- **v1.2.0**：集成 LLM 提示词优化功能
- **v1.3.0**：优化 API 调用流程，提升稳定性

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本扩展！

## 联系方式

如有问题或建议，请在 GitHub 仓库中提交 Issue。