# Seedream API Refiner Extension

中文readme详见[README_cn.md](README_cn.md)

Seedream API Refiner is a Stable Diffusion webUI extension that uses Seedream API to intelligently optimize generated images, enhancing image quality and detail representation.

## Key Features

- **Intelligent Image Optimization**: Uses Seedream API to optimize Stable Diffusion generated images with high quality
- **Prompt Translation & Optimization**: Built-in LLM translation function automatically translates and optimizes Chinese prompts to English for better results
- **Reference Image Support**: Supports uploading reference images to guide the optimization process and maintain style consistency
- **Advanced Image Preprocessing**: Integrates multiple image preprocessing tools (Canny, Depth, OpenPose, Lineart, etc.) for reference image processing
- **Flexible Control Units**: Provides 4 independent control units, similar to ControlNet usage
- **Automatic Output Management**: Optimized images are automatically saved to date-organized dedicated directories

## Technical Architecture

- **Core Functionality**: Calls Seedream API through OpenAI-compatible API interface
- **User Interface**: Builds intuitive operation interface using Gradio
- **Image Processing**: Integrates multiple preprocessors supporting edge detection, depth estimation, etc.
- **Extension Integration**: Seamlessly integrates into Stable Diffusion webUI's extension system

## Installation

1. **Clone Repository**: Clone this repository to your extensions directory
   ```bash
   git clone https://github.com/yourusername/sd-SeedreamAPI.git extensions/sd-SeedreamAPI
   ```

2. **Restart webUI**: Restart Stable Diffusion webUI to load the extension

3. **Install Dependencies**: The extension will automatically check and install required dependencies (like OpenAI library)

## Usage Guide

### Basic Usage

1. **Enable Extension**: Find the "Seedream API Refiner" section in the webUI interface and check the "Enable Seedream API Refiner" checkbox

2. **Configure API Key**: Find the "Seedream API Refiner" section in webUI settings and enter your Seedream API Key

3. **Set Prompt**: Enter optimization prompt in the "Seedream Prompt" text box
   - If not entered, it will use the webUI's positive prompt by default
   - If webUI positive prompt is also empty, it will use the default prompt: "Optimize this image, add more details and hires"

4. **Translate & Optimize Prompt**: Click the "Translate&Optimize" button to automatically translate and optimize the prompt using LLM

5. **Generate Images**: Create images as usual with Stable Diffusion

6. **View Results**: Optimized images will be automatically saved to `outputs-SeedreamRefiner/yy-mm-dd` directory

### Advanced Feature: Reference Images

1. **Expand Upload Area**: Click "Upload Images" to expand reference image settings

2. **Enable Reference Images**: Check the "Enable Reference Images" checkbox

3. **Upload Reference Images**: Click the "Reference Images" upload button, maximum 5 reference images can be uploaded

### Advanced Feature: Control Units

1. **Expand Image Preprocessing**: Click "Image Preprocessing" under "Upload Images" to expand control unit settings

2. **Configure Control Units**: Each control unit includes the following settings:
   - **Enable**: Enable this control unit
   - **Image**: Upload reference image
   - **Preprocessor Category**: Select preprocessor category (All, Hard Edge, Depth, Pose, Line)
   - **Preprocessor**: Select specific preprocessor (none, canny, depth_midas, openpose, lineart, lineart_anime)
   - **Enable Preview**: Enable preprocessing result preview
   - **Preview**: Click button to view preprocessing effect

3. **Use Control Units**: Enabled control units will pass preprocessed images as reference images to Seedream API

## Configuration Options

### Main Settings

- **Enable Seedream API Refiner**: Toggle to enable or disable the extension
- **Seedream Prompt**: Prompt for Seedream API to optimize the image
- **Translate&Optimize**: Button to translate and optimize prompt

### Reference Image Settings

- **Enable Reference Images**: Toggle to enable or disable reference images
- **Reference Images**: Upload reference images (maximum 5)

### Control Unit Settings

- **Enable**: Toggle to enable or disable control unit
- **Image**: Upload control image
- **Preprocessor Category**: Preprocessor category
- **Preprocessor**: Specific preprocessor
- **Enable Preview**: Enable preview
- **Preview**: Preview button

### Global Settings (webUI Settings Page)

- **Seedream API Key**: Your Seedream API Key (required)
- **Output Directory**: Directory where optimized images will be saved

## Preprocessor Description

| Preprocessor | Description | Application Scenario |
|-------------|-------------|----------------------|
| none | No preprocessing, use original image directly | Maintain original style |
| canny | Edge detection, extract image edges | Enhance contours and structure |
| depth_midas | Depth estimation, generate depth map | Enhance spatial sense and three-dimensionality |
| openpose | Pose detection, extract human pose | Maintain character pose consistency |
| lineart | Line extraction, generate line art | Enhance line art effects |
| lineart_anime | Anime style line extraction | Suitable for anime style images |

## Notes

- **API Key Required**: A valid Seedream API Key is required to use this extension
- **Processing Time**: Optimization process may take some time depending on API response speed
- **Reference Images**: Maximum 4 reference images are supported (including images processed by control units)
- **Output Path**: Optimized images will not be displayed in the webUI main interface, please check the specified output directory
- **Preprocessing**: Preprocessor operations will be executed on CPU to avoid occupying GPU resources
- **Prompt Optimization**: Using the "Translate&Optimize" function can achieve better optimization results

## Troubleshooting

- **API Key Error**: Please ensure you have entered a valid Seedream API Key
- **No Output**: Check if the output directory exists and has write permissions
- **Slow Processing**: This may be due to long API response times, please be patient
- **Preprocessing Failure**: Check if the image format is correct and ensure the image is not empty
- **Invalid Reference Images**: Ensure uploaded reference images are clear and relevant to the optimization goal

## System Requirements

- Stable Diffusion webUI
- Python 3.10+
- Internet connection (for calling Seedream API)
- Sufficient disk space (for saving optimized images)

## Example Workflows

1. **Basic Optimization**: Enable extension → Enter prompt → Generate image → Automatic optimization

2. **Style Reference**: Enable extension → Upload style reference image → Generate image → Style-based optimization

3. **Structure Control**: Enable extension → Configure Canny control unit → Upload edge reference → Generate image → Structure-preserving optimization

4. **Pose Maintenance**: Enable extension → Configure OpenPose control unit → Upload pose reference → Generate image → Pose-preserving optimization

## License

MIT License

## Changelog

- **v1.0.0**: Initial version, implementing basic functionality
- **v1.1.0**: Added control unit functionality, supporting multiple preprocessors
- **v1.2.0**: Integrated LLM prompt optimization functionality
- **v1.3.0**: Optimized API calling process, improved stability

## Contribution

Welcome to submit Issues and Pull Requests to improve this extension!

## Contact

For questions or suggestions, please submit an Issue in the GitHub repository.