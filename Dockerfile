#FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
#FROM cc0cc75bbf67
FROM 700ac2691c19
# 配置程序依赖环境
RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential \
 cmake \
 curl \
 ca-certificates \
 libjpeg-dev \
 libpng-dev && \
 rm -rf /var/lib/apt/lists/*
# 安装imageio
#RUN pip install imageio

# 安装imageio
#RUN pip install imageio
# 安装matplotlib
#RUN conda install matplotlib
# 安装yacs和termcolor
RUN pip install Pillow
# 安装mmcv
#RUN pip install mmcv
# 安装timm
#RUN pip install timm==0.3.2
# 安装pytorch
#RUN pip3 install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# 将程序复制容器内，表示在/workspace 路径下
COPY . /workspace
# 确定容器启动时程序运行路径
WORKDIR /workspace
# 确定容器启动命令。以 python 示例，python 表示编译器，run.py 表示执
# 行文件，/input_path 和/output_path 为容器内绝对路径，测评时会自动将
# 测试数据挂载到容器内/input_path 路径，无需修改
CMD ["python", "run.py", "/input_path", "/output_path"]