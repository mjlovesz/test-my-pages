# ait profile 功能使用指南

## 简介
- 一键式全流程推理工具ait集成了Profiling性能分析工具，用于分析运行在昇腾AI处理器上的APP工程各个运行阶段的关键性能瓶颈并提出针对性能优化的建议，最终实现产品的极致性能。
- Profiling数据通过二进制可执行文件”msprof”进行数据采集，使用该方式采集Profiling数据需确保应用工程或算子工程所在运行环境已安装Toolkit组件包。
- 该工具使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/atlasprofiling_16_0004.html)

## 工具安装
- 工具安装请见 [ait一体化工具使用指南](../../README.md)

## 使用方法
### 功能介绍
#### 使用入口
profile可以直接通过ait命令行形式启动模型推理的性能分析。使用ait benchmark(ait benchmark为ait自带的推理工程，用户只需修改om路径即可进行模型推理的性能分析及数据采集)推理的性能分析的命令如下：
```bash
ait profile --application "ait benchmark -om *.om --device 0" --output <some path>
```
其中，*为OM离线模型文件名；<some path>为路径名称。
得到主要输出结果如下：
```
<some path>
└── profiler
    └── PROF_000001_20231023172400639_NJDOONIBJCPMJGGB
        ├── device_0 # device侧的结果，device_0表示device id 为0的芯片的性能数据
        │   ├── data # 原始性能数据
        │   ├── log # profiling过程的log日志
        │   ├── summary # 性能数据汇总表格
        │   └── timeline # 通过时间轴呈现性能数据
        └── host
            ├── data # host侧的原始数据
            ├── log # profiling过程的log日志
            ├── summary # 性能数据汇总表格
            └── timeline # 通过时间轴呈现性能数据

```
summary 和 timeline中的文件因命令行参数的选择而不同，请依据[使用场景](#使用场景)查看每个文件的具体含义。
- 采集AI任务运行性能数据相关[summary文件和timeline文件介绍](https://gitee.com/ascend/ait/tree/master/ait/examples/cli/profile/02_collect_ai_task_data)
- 采集昇腾AI处理器系统数据相关[summary文件和timeline文件介绍](https://gitee.com/ascend/ait/tree/master/ait/examples/cli/profile/03_collect_ascend_ai_processor_data)
summary文件夹下的文件可以用Excel等软件打开，timeline下的json文件可以在chrome浏览器下用[tracing](chrome://tracing)打开。


#### 参数说明
  | 参数名                    | 描述                                       | 必选   |
  | ------------------------ | ---------------------------------------- | ---- |
  | --application            | 配置为运行环境上app可执行文件，可配置ait自带的benchmark推理程序，application带参数输入，此时需要使用英文双引号将”application”的参数值括起来，例如--application "ait benchmark -om /home/HwHiAiUser/resnet50.om"，用户使用仅需修改指定om路径 | 是    |
  | -o, --output             | 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录                                                                | 否    |
  | --model-execution        | 控制ge model execution性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --sys-hardware-mem       | 控制DDR，LLC的读写带宽数据采集开关，可选on或off，默认为on。 | 否    |
  | --sys-cpu-profiling      | CPU（AI CPU、Ctrl CPU、TS CPU）采集开关。可选on或off，默认值为off。                           | 否    |
  | --sys-profiling          | 系统CPU usage及System memory采集开关。可选on或off，默认值为off。 | 否    |
  | --sys-pid-profiling      | 进程的CPU usage及进程的memory采集开关。可选on或off，默认值为off。 | 否    |
  | --dvpp-profiling         | DVPP采集开关，可选on或off，默认值为on | 否    |
  | --runtime-api            | 控制runtime api性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --task-time              | 控制ts timeline数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --aicpu                  | aicpu开关，可选on或off，默认为on。 | 否  |
  | -h, --help               | 工具使用帮助信息。               | 否  |

  ### 使用场景
请移步[profile使用示例](../../examples/cli/profile/)
  | 使用示例               | 使用场景                                 |
  |-----------------------| ---------------------------------------- |
  | [01_basic_usage](../../examples/cli/profile/01_basic_usage)    | 基础示例，对benchmark推理om模型执行性能分析       |
  | [02_collect_ai_task_data](../../examples/cli/profile/02_collect_ai_task_data) | 采集AI任务运行性能数据 |
  | [03_collect_ascend_ai_processor_data](../../examples/cli/profile/03_collect_ascend_ai_processor_data) | 采集昇腾AI处理器系统数据 |

  ### 性能分析实践案例
应用ait profile 进行性能分析可以参考案例:[基于ait的性能调优案例](https://gitee.com/ascend/ait/wikis/%E6%A1%88%E4%BE%8B%E5%88%86%E4%BA%AB/%E5%9F%BA%E4%BA%8Eait%E7%9A%84%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98%E6%A1%88%E4%BE%8B)