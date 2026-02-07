import gradio as gr
import requests
import os
import time
from datetime import datetime
from modules import scripts, script_callbacks, shared
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError:
    print("[Seedream API Refiner] OpenAI library not found, installing...")
    import subprocess
    subprocess.run(["pip", "install", "openai"], check=True)
    from openai import OpenAI

def call_seedream_llm_api(api_key, prompt, task_type):
    """调用 Seedream LLM API 进行翻译和优化"""
    print("[Seedream API Refiner] Starting call_seedream_llm_api")
    try:
        # 使用OpenAI兼容的API接口
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key
        )
        
        # 根据任务类型构建系统提示
        if task_type == "translate":
            system_prompt = "You are a professional English translator and copywriter. Please translate the following Chinese text into English, and optimize it to make it more suitable for image generation prompts. Make the translation natural and fluent, and ensure that the optimized prompt can effectively guide the image generation model."
        else:
            system_prompt = "You are a professional copywriter specializing in image generation prompts. Please optimize the following prompt to make it more effective for guiding image generation models."
        
        # 调用LLM API
        response = client.chat.completions.create(
            model="deepseek-v3-2-251201",  # 使用通用模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # 获取优化后的提示词
        optimized_prompt = response.choices[0].message.content.strip()
        print(f"[Seedream API Refiner] LLM API response: {optimized_prompt}")
        return optimized_prompt
    except Exception as e:
        print(f"[Seedream API Refiner] Error in call_seedream_llm_api: {e}")
        import traceback
        traceback.print_exc()
        return prompt

class Script(scripts.Script):
    _initialized = False
    
    def title(self):
        if not Script._initialized:
            print("[Seedream API Refiner] Initializing extension")
            Script._initialized = True
        return "Seedream API Refiner"
    
    def show(self, is_img2img):
        print(f"[Seedream API Refiner] Showing extension for {'img2img' if is_img2img else 'txt2img'}")
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        print(f"[Seedream API Refiner] Creating UI for {'img2img' if is_img2img else 'txt2img'}")
        # 从设置中加载值
        enabled_value = getattr(shared.opts, "seedream_enabled", False)
        seedream_prompt_value = getattr(shared.opts, "seedream_prompt", "Optimize this image, add more details and hires")
        output_dir_value = getattr(shared.opts, "seedream_output_dir", "outputs_SeedreamRefiner")
        print(f"[Seedream API Refiner] Loaded settings: enabled={enabled_value}, output_dir={output_dir_value}")
        
        # 创建主菜单
        with gr.Accordion("Seedream API Refiner", open=False, elem_id="seedream_api_refiner") as main_accordion:
            with gr.Row():
                enabled = gr.Checkbox(value=enabled_value, label="Enable Seedream API Refiner", elem_id="seedream_enabled")
            
            with gr.Row():
                seedream_prompt = gr.Textbox(value=seedream_prompt_value, label="Seedream Prompt", elem_id="seedream_prompt", lines=3, placeholder="若不输入，默认使用webUI中的正面提示词")
            
            with gr.Row():
                # 创建按钮
                translate_optimize_button = gr.Button("Translate&Oprimize", elem_id="seedream_translate_optimize_button")
                
                # 添加鼠标悬停提示（使用Gradio的tooltip参数）
                translate_optimize_button.tooltip = "使用LLM翻译并优化Seedream提示词"
            
            # 创建参考图片子菜单
            with gr.Accordion("Upload Images", open=False) as upload_accordion:
                upload_enabled = gr.Checkbox(value=False, label="Enable Reference Images", elem_id="seedream_upload_enabled")
                reference_images = gr.Files(label="Reference Images (max 5)", elem_id="seedream_reference_images", file_count="multiple", file_types=["image"])
                
                # 创建图像预处理子菜单（在Upload Images下）
                with gr.Accordion("Image Preprocessing", open=False) as preprocessing_accordion:
                    # 创建5个控制单元tab，类似ControlNet的设计
                    with gr.Tabs(label="Control Units") as control_tabs:
                        # 定义预处理器分类选项（仅保留需要的分类）
                        preprocessor_categories = [
                            "全部",
                            "硬边缘(Canny)",
                            "深度(Depth)",
                            "姿态(OpenPose)",
                            "线条(Lineart)"
                        ]
                        
                        # 定义预处理器选项（仅保留需要的预处理器）
                        preprocessor_options = [
                            "none",
                            "canny",
                            "depth_midas",
                            "openpose",
                            "lineart",
                            "lineart_anime"
                        ]
                        
                        # 创建4个控制单元
                        tab_components = []
                        for i in range(4):
                            with gr.Tab(label=f"Control Unit {i}") as tab:
                                # 每个控制单元内部的结构
                                # 只保留单张图像上传，删除batch processing和multi-inputs
                                # 图像上传区域
                                unit_image = gr.Image(label="Image", elem_id=f"seedream_unit_{i}_image", interactive=True, height=242)
                                
                                # 预处理结果预览区域
                                unit_preview = gr.Image(label="Preprocessor Preview", elem_id=f"seedream_unit_{i}_preview", interactive=False, height=242, visible=True)
                                
                                # 控制选项
                                with gr.Row():
                                    unit_enabled = gr.Checkbox(value=False, label="Enable", elem_id=f"seedream_unit_{i}_enabled")
                                    unit_preview_enabled = gr.Checkbox(value=False, label="Enable Preview", elem_id=f"seedream_unit_{i}_preview_enabled")
                                
                                # 预处理与模型筛选
                                with gr.Accordion("Preprocessor Selection", open=True) as selection_accordion:
                                    # 预处理分类筛选（使用radio buttons，类似ControlNet的布局）
                                    with gr.Row():
                                        unit_preprocessor_category = gr.Radio(preprocessor_categories, label="Preprocessor Category", value="全部", elem_id=f"seedream_unit_{i}_preprocessor_category", interactive=True)
                                    
                                    # 预处理器细选（使用dropdown）
                                    with gr.Row():
                                        unit_preprocessor = gr.Dropdown(preprocessor_options, label="Preprocessor", value="none", elem_id=f"seedream_unit_{i}_preprocessor")
                                        unit_preview_button = gr.Button("Preview", elem_id=f"seedream_unit_{i}_preview_button")
                                
                                # 添加组件到tab_components
                                tab_components.extend([
                                    unit_enabled, 
                                    unit_image, 
                                    unit_preprocessor_category, 
                                    unit_preprocessor, 
                                    unit_preview, 
                                    unit_preview_button,
                                    unit_preview_enabled
                                ])
                                
                                # 预处理器分类与具体预处理器的映射
                                category_to_preprocessors = {
                                    "全部": preprocessor_options,
                                    "硬边缘(Canny)": ["none", "canny"],
                                    "深度(Depth)": ["none", "depth_midas"],
                                    "姿态(OpenPose)": ["none", "openpose"],
                                    "线条(Lineart)": ["none", "lineart", "lineart_anime"]
                                }
                                
                                # 绑定预处理器分类选择事件
                                def update_preprocessors(category):
                                    # 根据选择的分类获取对应的预处理器选项
                                    filtered_preprocessors = category_to_preprocessors.get(category, preprocessor_options)
                                    # 更新预处理器下拉菜单
                                    return gr.Dropdown.update(choices=filtered_preprocessors, value="none")
                                
                                unit_preprocessor_category.change(fn=update_preprocessors, inputs=[unit_preprocessor_category], outputs=[unit_preprocessor])
                                
                                # 绑定预览按钮点击事件
                                def preview_image(image, preprocessor, preview_enabled):
                                    # 检查图像是否有效
                                    if image is None:
                                        return None
                                    # 检查预览是否启用
                                    if not preview_enabled:
                                        return None
                                    # 检查图像是否为空数组
                                    import numpy as np
                                    if isinstance(image, np.ndarray):
                                        if image.size == 0:
                                            return None
                                    # 调用预处理函数
                                    processed_image = self.preprocess_image(image, preprocessor)
                                    return processed_image
                                
                                unit_preview_button.click(fn=preview_image, inputs=[unit_image, unit_preprocessor, unit_preview_enabled], outputs=[unit_preview])
                                
                                # 当启用预览时，自动显示预览区域
                                def update_preview_visibility(preview_enabled):
                                    return gr.Image.update(visible=preview_enabled)
                                
                                unit_preview_enabled.change(fn=update_preview_visibility, inputs=[unit_preview_enabled], outputs=[unit_preview])
        
        # 保存设置
        def save_settings(enabled_val, seedream_prompt_val, upload_enabled_val, *unit_vals):
            # 保存设置
            shared.opts.save(shared.config_filename)
            return None
        
        # 当设置改变时保存
        enabled.change(fn=save_settings, inputs=[enabled, seedream_prompt, upload_enabled] + tab_components)
        seedream_prompt.change(fn=save_settings, inputs=[enabled, seedream_prompt, upload_enabled] + tab_components)
        upload_enabled.change(fn=save_settings, inputs=[enabled, seedream_prompt, upload_enabled] + tab_components)
        # 为每个tab的组件添加change事件，但跳过Button组件（Button没有change方法）
        for comp in tab_components:
            if hasattr(comp, 'change'):
                comp.change(fn=save_settings, inputs=[enabled, seedream_prompt, upload_enabled] + tab_components)
        
        # 翻译并优化提示词的处理函数
        def translate_optimize_prompt(prompt):
            print(f"[Seedream API Refiner] Starting translate_optimize_prompt with prompt: {prompt}")
            try:
                # 优先使用设置中的API key
                api_key = getattr(shared.opts, "seedream_api_key", "")
                if not api_key:
                    print("[Seedream API Refiner] API key not found, using empty string")
                
                # 调用LLM API进行翻译和优化
                optimized_prompt = call_seedream_llm_api(api_key, prompt, "translate")
                print(f"[Seedream API Refiner] Translated and optimized prompt: {optimized_prompt}")
                
                # 注意：不再保存到设置，因为我们已经删除了seedream_prompt设置项
                # 直接返回优化后的提示词，让Gradio更新输入框
                return optimized_prompt
            except Exception as e:
                print(f"[Seedream API Refiner] Error in translate_optimize_prompt: {e}")
                import traceback
                traceback.print_exc()
                return prompt
        
        # 绑定按钮点击事件
        translate_optimize_button.click(
            fn=translate_optimize_prompt,
            inputs=[seedream_prompt],
            outputs=[seedream_prompt],
            show_progress=True
        )
        
        # 预览功能已经在每个控制单元内部绑定，这里不再重复绑定
        
        self.infotext_fields = (
            (enabled, "Seedream API Refiner enabled"),
            (seedream_prompt, "Seedream Prompt"),
            (upload_enabled, "Seedream Reference Images enabled"),
        )
        
        # 添加每个tab的组件到infotext_fields
        for i in range(4):
            if i*7 + 3 < len(tab_components):
                unit_enabled = tab_components[i*7]
                unit_preprocessor = tab_components[i*7 + 3]
                self.infotext_fields += (
                    (unit_enabled, f"Seedream Unit {i+1} enabled"),
                    (unit_preprocessor, f"Seedream Unit {i+1} preprocessor"),
                )
        
        print(f"[Seedream API Refiner] UI created successfully for {'img2img' if is_img2img else 'txt2img'}")
        return [enabled, seedream_prompt, upload_enabled, reference_images] + tab_components
    
    def postprocess_image(self, p, processed, enabled, seedream_prompt, upload_enabled, reference_images=None, *tab_components):
        print("[Seedream API Refiner] Starting postprocess_image")
        try:
            print(f'[Seedream API Refiner] my.shared.opts={shared.opts}')
            # 优先使用设置中的值
            api_key = getattr(shared.opts, "seedream_api_key", "")
            #seedream_prompt = getattr(shared.opts, "seedream_prompt", seedream_prompt)
            output_dir = getattr(shared.opts, "seedream_output_dir", "outputs-SeedreamRefiner")
            
            # 解析tab组件（每个控制单元有7个组件）
            unit_data = []
            for i in range(4):
                if i*7 + 6 < len(tab_components):
                    unit_enabled = tab_components[i*7]
                    unit_image = tab_components[i*7 + 1]
                    unit_preprocessor = tab_components[i*7 + 3]  # 预处理器在第4个位置
                    unit_data.append((unit_enabled, unit_image, unit_preprocessor))
            
            print(f"[Seedream API Refiner] Postprocess settings: enabled={enabled}, output_dir={output_dir}")
            print(f"[Seedream API Refiner] Seedream Prompt: {seedream_prompt}")
            print(f"[Seedream API Refiner] Reference images: {len(reference_images) if reference_images else 0}")
            print(f"[Seedream API Refiner] Processing units: {len([u for u in unit_data if u[0]])}")
            
            if not enabled:
                print("[Seedream API Refiner] Extension is disabled, skipping postprocessing")
                return
            
            else:
                print("[Seedream API Refiner] Extension is enabled, proceeding with postprocessing")

            if not api_key:
                print("[Seedream API Refiner] API key is required")
                return
            
            # 创建输出目录
            date_str = datetime.now().strftime("%y-%m-%d")
            full_output_dir = os.path.join(output_dir, date_str)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # 注意：用户上传的参考图片不应该显示在输出图像区域
            # 限制最多4张参考图片
            if reference_images:
                reference_images = reference_images[:4]
            
            # 处理新的tab组件中的图像
            processed_images = []
            
            # 处理参考图片
            if reference_images:
                processed_images.extend(reference_images)
                print(f"[Seedream API Refiner] Added {len(reference_images)} user-uploaded reference images")
            
            # 处理tab组件中的图像
            for i, (unit_enabled, unit_image, unit_preprocessor) in enumerate(unit_data):
                try:
                    # 安全检查unit_enabled和unit_image
                    is_enabled = False
                    has_image = False
                    
                    # 检查unit_enabled
                    if isinstance(unit_enabled, bool):
                        is_enabled = unit_enabled
                    elif hasattr(unit_enabled, 'any'):
                        is_enabled = bool(unit_enabled.any())
                    
                    # 检查unit_image
                    if unit_image is not None:
                        if not isinstance(unit_image, (list, tuple)):
                            has_image = True
                        elif len(unit_image) > 0:
                            has_image = True
                    
                    if is_enabled and has_image:
                        # 处理图像
                        processed_image = self.preprocess_image(unit_image, unit_preprocessor)
                        if processed_image:
                            processed_images.append(processed_image)
                            print(f"[Seedream API Refiner] Added processed image from Unit {i+1}")
                except Exception as e:
                    print(f"[Seedream API Refiner] Error processing image from Unit {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 限制最多4张参考图片
            all_reference_images = processed_images[:4]
            print(f"[Seedream API Refiner] Total reference images: {len(all_reference_images)}")
            
            # 获取webUI中的正面提示词
            positive_prompt = ""
            if hasattr(p, 'prompt') and p.prompt:
                positive_prompt = p.prompt
            
            # 确定使用的提示词
            final_prompt = seedream_prompt
            if not final_prompt:
                if positive_prompt:
                    final_prompt = positive_prompt
                else:
                    final_prompt = "Optimize this image, add more details and hires"
            
            print(f"[Seedream API Refiner] Final Prompt: {final_prompt}")
            
            # 处理图像 - 适配不同版本的参数结构
            images = []
            if hasattr(processed, 'images'):
                # 旧版本：processed是一个包含images属性的对象
                images = processed.images
            elif hasattr(processed, 'image'):
                # 新版本：processed是PostprocessImageArgs，包含单个image属性
                images = [processed.image]
            
            if not images:
                print("[Seedream API Refiner] No images to process")
                return
            
            # 抓取本次SD输出的图像作为参考图
            # 限制最多4张参考图片
            sd_output_images = images[:4]
            print(f"[Seedream API Refiner] Using {len(sd_output_images)} SD output images as reference")
            
            # 处理每张生成的图像
            for i, img in enumerate(images):
                try:
                    # 调用 Seedream API 进行优化
                    print(f"[Seedream API Refiner] Processing image {i+1}")
                    # 将SD输出的图像作为参考图
                    reference_images = all_reference_images.copy() if all_reference_images else []
                    # 添加当前SD输出的所有图像作为参考
                    for j, ref_img in enumerate(sd_output_images):
                        reference_images.append(ref_img)
                    # 限制最多4张参考图片
                    reference_images = reference_images[:4]
                    print(f"[Seedream API Refiner] Total reference images: {len(reference_images)}")
                    optimized_img = self.call_seedream_api(img, api_key, final_prompt, reference_images)
                    
                    #print(f'[Seedream API Refiner] my.optimized_img.length=1024x1024')

                    if optimized_img:
                        # 保存优化后的图像
                        seed = getattr(p, 'seed', 0)
                        filename = f"seedream_{seed}_{i}_{int(time.time())}.png"
                        filepath = os.path.join(full_output_dir, filename)

                        print(f"[Seedream API Refiner] my.filepath= {filepath}")
                        
                        try:
                            optimized_img.save(filepath)
                            print(f"[Seedream API Refiner] Optimized image saved to: {filepath}")
                            
                            # 适配不同版本的输出方式
                            # 确保processed对象有images属性
                            if not hasattr(processed, 'images'):
                                processed.images = []
                            
                            # 添加优化后的图像到输出列表
                            processed.images.append(optimized_img)
                            print(f"[Seedream API Refiner] Added optimized image to processed.images, total: {len(processed.images)}")
                            
                            # 确保processed对象有infotexts属性
                            if not hasattr(processed, 'infotexts'):
                                processed.infotexts = []
                            
                            # 添加信息到图像信息中
                            info_text = f", Seedream optimized: {final_prompt}"
                            if all_reference_images:
                                info_text += f", {len(all_reference_images)} reference images"
                            
                            # 根据现有infotexts的长度添加相应的信息
                            if i < len(processed.infotexts):
                                # 如果有对应的原始图像信息，添加到后面
                                processed.infotexts.append(processed.infotexts[i] + info_text)
                            else:
                                # 如果没有对应的原始图像信息，创建新的
                                processed.infotexts.append(f"Seedream optimized: {final_prompt}")
                            print(f"[Seedream API Refiner] Added info text to processed.infotexts, total: {len(processed.infotexts)}")
                            
                            # 对于新版本的webUI，可能需要更新其他属性
                            if hasattr(processed, 'index_of_first_image'):
                                print(f"[Seedream API Refiner] WebUI version with index_of_first_image, value: {processed.index_of_first_image}")
                        except Exception as save_error:
                            print(f"[Seedream API Refiner] Error saving optimized image: {save_error}")
                    else:
                        print(f"[Seedream API Refiner] Failed to optimize image {i+1}")
                        print("[Seedream API Refiner] Please check:")
                        print("1. Your API key is correct")
                        print("2. The API endpoint is reachable")
                        print("3. Your network connection is stable")
                except Exception as e:
                    print(f"[Seedream API Refiner] Error processing image: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[Seedream API Refiner] Error in postprocess_image: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[Seedream API Refiner] Postprocess_image completed")
    
    def preprocess_image(self, image, preprocessor):
        """对图像进行预处理"""
        print(f"[Seedream API Refiner] Starting preprocess_image with preprocessor: {preprocessor}")
        try:
            from PIL import Image
            import numpy as np
            import sys
            import os
            import torch
            
            # 添加扩展根目录到系统路径的最前面，确保优先使用本地的annotator模块
            extension_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 移除可能存在的其他annotator路径
            sys.path = [p for p in sys.path if 'annotator' not in p]
            # 将扩展根目录添加到最前面
            if extension_root not in sys.path:
                sys.path.insert(0, extension_root)
            
            # 确保图像是PIL Image对象
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return None
            
            # 如果选择了"none"预处理器，直接返回原始图像
            if preprocessor == "none":
                return image
            
            # 转换为numpy数组进行处理
            img_array = np.array(image)
            
            # 确保图像是RGB格式
            if img_array.ndim == 2:
                # 灰度图像，转换为RGB
                import cv2
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                # RGBA图像，转换为RGB
                img_array = img_array[:, :, :3]
            
            # 强制使用CPU进行预处理器运算
            print("[Seedream API Refiner] Forcing preprocessor to use CPU")
            # 保存原始的devices模块
            import modules
            original_devices = modules.devices
            # 创建一个临时的devices模块，始终返回CPU
            class TempDevices:
                def get_device_for(self, *args, **kwargs):
                    return torch.device('cpu')
            # 替换devices模块
            modules.devices = TempDevices()
            
            try:
                # 根据预处理器类型进行处理
                processed_array = None
                
                if preprocessor == "canny":
                    # 使用本地annotator的canny预处理器
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("canny", os.path.join(extension_root, "annotator", "canny", "__init__.py"))
                    canny_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(canny_module)
                    edges = canny_module.apply_canny(img_array, 100, 200)
                    processed_array = np.stack([edges, edges, edges], axis=2)  # 转换为RGB
                elif preprocessor == "lineart":
                    # 使用本地annotator的lineart预处理器
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("lineart", os.path.join(extension_root, "annotator", "lineart", "__init__.py"))
                    lineart_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(lineart_module)
                    detector = lineart_module.LineartDetector('sk_model.pth')
                    line = detector(img_array)
                    processed_array = np.stack([line, line, line], axis=2)  # 转换为RGB
                    detector.unload_model()  # 卸载模型以释放内存
                elif preprocessor == "lineart_anime":
                    # 使用本地annotator的lineart_anime预处理器
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("lineart_anime", os.path.join(extension_root, "annotator", "lineart_anime", "__init__.py"))
                    lineart_anime_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(lineart_anime_module)
                    detector = lineart_anime_module.LineartAnimeDetector()
                    line = detector(img_array)
                    processed_array = np.stack([line, line, line], axis=2)  # 转换为RGB
                    detector.unload_model()  # 卸载模型以释放内存
                elif preprocessor == "depth_midas":
                    # 使用本地annotator的midas预处理器
                    import importlib.util
                    # 确保midas目录在系统路径中
                    midas_dir = os.path.join(extension_root, "annotator", "midas")
                    if midas_dir not in sys.path:
                        sys.path.insert(0, midas_dir)
                    # 导入midas模块
                    spec = importlib.util.spec_from_file_location("midas", os.path.join(extension_root, "annotator", "midas", "__init__.py"))
                    midas_module = importlib.util.module_from_spec(spec)
                    # 添加模块到sys.modules，以便相对导入正常工作
                    sys.modules["midas"] = midas_module
                    spec.loader.exec_module(midas_module)
                    depth, _ = midas_module.apply_midas(img_array)  # 获取第一个返回值
                    processed_array = np.stack([depth, depth, depth], axis=2)  # 转换为RGB
                    midas_module.unload_midas_model()  # 卸载模型以释放内存
                elif preprocessor == "openpose":
                    # 使用本地annotator的openpose预处理器
                    import importlib.util
                    # 确保openpose目录在系统路径中
                    openpose_dir = os.path.join(extension_root, "annotator", "openpose")
                    if openpose_dir not in sys.path:
                        sys.path.insert(0, openpose_dir)
                    # 导入openpose模块
                    spec = importlib.util.spec_from_file_location("openpose", os.path.join(extension_root, "annotator", "openpose", "__init__.py"))
                    openpose_module = importlib.util.module_from_spec(spec)
                    # 添加模块到sys.modules，以便相对导入正常工作
                    sys.modules["openpose"] = openpose_module
                    spec.loader.exec_module(openpose_module)
                    detector = openpose_module.OpenposeDetector()
                    pose = detector(img_array)
                    processed_array = pose
                    detector.unload_model()  # 卸载模型以释放内存
                else:
                    # 对于其他预处理器，返回原始图像
                    print(f"[Seedream API Refiner] Preprocessor {preprocessor} not implemented, returning original image")
                    return image
                
                # 将处理后的数组转换回PIL Image
                if processed_array is not None:
                    # 确保数组是uint8类型
                    if processed_array.dtype != np.uint8:
                        processed_array = processed_array.astype(np.uint8)
                    # 确保数组维度正确
                    if len(processed_array.shape) == 2:
                        processed_array = np.stack([processed_array, processed_array, processed_array], axis=2)
                    elif processed_array.shape[2] == 1:
                        processed_array = np.concatenate([processed_array, processed_array, processed_array], axis=2)
                    processed_image = Image.fromarray(processed_array)
                    print(f"[Seedream API Refiner] Preprocessing completed successfully with {preprocessor}")
                    return processed_image
                else:
                    print(f"[Seedream API Refiner] Preprocessing failed or returned no result for {preprocessor}")
                    return image
            finally:
                # 恢复原始的devices模块
                modules.devices = original_devices
                print("[Seedream API Refiner] Restored original devices module")
        except Exception as e:
            print(f"[Seedream API Refiner] Error in preprocess_image: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def call_seedream_api(self, img, api_key, seedream_prompt, reference_images=None):
        """调用 Seedream API 进行图像优化"""
        print("[Seedream API Refiner] Starting call_seedream_api")
        try:
            # 处理参考图像
            reference_images_data = []
            if reference_images:
                print(f"[Seedream API Refiner] Processing {len(reference_images)} reference images")
                for i, ref_img in enumerate(reference_images[:4]):
                    try:
                        if hasattr(ref_img, 'save'):
                            # 如果是图像对象
                            ref_buffer = io.BytesIO()
                            ref_img.save(ref_buffer, format="PNG")
                            ref_buffer.seek(0)
                            reference_images_data.append(ref_buffer.getvalue())
                            print(f"[Seedream API Refiner] Added reference image {i+1}")
                    except Exception as e:
                        print(f"[Seedream API Refiner] Error processing reference image {i+1}: {e}")
            
            # 使用OpenAI兼容的API接口
            print("[Seedream API Refiner] Initializing OpenAI client")
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=api_key
            )
            
            print("[Seedream API Refiner] Calling images.generate API")
            print(f"[Seedream API Refiner] Prompt: {seedream_prompt}")
            
            # 调用图像生成API
            # 这里我们将参考图像的信息添加到提示词中
            if reference_images_data:
                #seedream_prompt += f"\n\nGuidance: These images provided as reference for style, composition, lighting, shadows,and texture.You can add more detiled shadows or decorations.The first one in reference images is main image you need to oprimize.If there are openpose or depth or linart images, just follow them. Please use these images as a reference to enhance the visual effect, e.g. improved lighting, shadows, fix bad hands or arms.Do NOT change style of the reference image, e.g. change anime to real photo.Fix text problems in the output image if cuold. You need follow the prompt above the guide than these guidance."
                #print(f"[Seedream API Refiner] Added reference image information to prompt")
                pass     
            else:
                print(f"[Seedream API Refiner] No reference images processed")
            
            print(f"[Seedream API Refiner] my.final prompt: {seedream_prompt}")

            # 准备API调用参数
            api_params = {
                "model": "doubao-seedream-4-5-251128",
                "prompt": seedream_prompt,
                "response_format": "url",
                "size": "4K",
                
            }
            
            # 如果有参考图像，添加到extra_body参数中
            if reference_images and len(reference_images) > 0:
                # 选择第一张参考图像进行处理
                ref_img = reference_images[0]
                try:
                    # 检查ref_img是否有save方法
                    if hasattr(ref_img, 'save'):
                        # 保存参考图像到临时文件
                        import tempfile
                        import base64
                        
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # 保存图像
                        ref_img.save(temp_path, format="PNG")
                        print(f"[Seedream API Refiner] Saved reference image to {temp_path}")
                        
                        # 读取并编码图像为base64
                        with open(temp_path, "rb") as f:
                            image_data = f.read()
                        
                        # 将图像数据转换为base64字符串
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                        
                        # 添加extra_body参数
                        api_params["extra_body"] = {
                            "image": f"data:image/png;base64,{base64_image}",
                            "watermark": False
                        }
                        print("[Seedream API Refiner] Added reference image to extra_body")
                        
                        # 清理临时文件
                        import os
                        os.unlink(temp_path)
                        print(f"[Seedream API Refiner] Cleaned up temporary file {temp_path}")
                    else:
                        print(f"[Seedream API Refiner] Reference image is not a saveable object: {type(ref_img)}")
                except Exception as e:
                    print(f"[Seedream API Refiner] Error processing reference image: {e}")
            else:
                print(f"[Seedream API Refiner] No reference images to process")
            
            print("[Seedream API Refiner] Calling images.generate API with parameters")
            print(f"[Seedream API Refiner] Model: {api_params['model']}")
            print(f"[Seedream API Refiner] Size: {api_params['size']}")
            print(f"[Seedream API Refiner] Has extra_body: { 'extra_body' in api_params }")
            
            # 调用API
            resp = client.images.generate(**api_params)
            
            print("[Seedream API Refiner] API response received")
            print(f"[Seedream API Refiner] Response: {resp}")
            
            # 从响应中获取图像URL
            if resp.data and len(resp.data) > 0:
                image_url = resp.data[0].url
                print(f"[Seedream API Refiner] Image URL: {image_url}")
                
                # 下载生成的图像
                print("[Seedream API Refiner] Downloading generated image")
                image_response = requests.get(image_url, timeout=30, verify=False)
                if image_response.status_code == 200:
                    optimized_img = Image.open(io.BytesIO(image_response.content))
                    print("[Seedream API Refiner] Image downloaded successfully")
                    return optimized_img
                else:
                    print(f"[Seedream API Refiner] Failed to download image: {image_response.status_code}")
                    return None
            else:
                print("[Seedream API Refiner] No image data in response")
                return None
                
        except Exception as e:
            print(f"[Seedream API Refiner] Error in call_seedream_api: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            print("[Seedream API Refiner] call_seedream_api completed")

# 注册设置选项
def on_ui_settings():
    print("[Seedream API Refiner] Registering settings options")
    try:
        section = ('seedream_api', 'Seedream API Refiner')
         
        # 检查shared.opts是否存在
        if not hasattr(shared, 'opts'):
            print("[Seedream API Refiner] shared.opts not available")
            return
        
        # 检查add_option方法是否存在
        if not hasattr(shared.opts, 'add_option'):
            print("[Seedream API Refiner] shared.opts.add_option not available")
            return
        
        # 检查OptionInfo是否存在
        if not hasattr(shared, 'OptionInfo'):
            print("[Seedream API Refiner] shared.OptionInfo not available")
            return
        
        # 逐个添加选项，每个都有错误处理
        try:
            shared.opts.add_option(
                "seedream_api_key",
                shared.OptionInfo(
                    "",
                    "Seedream API Key",
                    gr.Textbox,
                    {"type": "password"},
                    section=section
                )
            )
        except Exception as e:
            print(f"[Seedream API Refiner] Error adding seedream_api_key: {e}")
        
        try:
            shared.opts.add_option(
                "seedream_output_dir",
                shared.OptionInfo(
                    "outputs/SeedreamRefiner",
                    "Output Directory",
                    gr.Textbox,
                    section=section
                )
            )
        except Exception as e:
            print(f"[Seedream API Refiner] Error adding seedream_output_dir: {e}")
            
        print("[Seedream API Refiner] Settings registration completed")
    except Exception as e:
        print(f"[Seedream API Refiner] Error in on_ui_settings: {e}")
        import traceback
        traceback.print_exc()

# 注册回调
script_callbacks.on_ui_settings(on_ui_settings)
