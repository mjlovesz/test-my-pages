# 大模型推理精度工具（Large Language Model Debug Tool)

## 简介

目前昇腾大模型推理框架主要有 [**加速库(atb)**](../glossary/README.md#at-Ascend-Transformer-Boost) 和 [**torchair**](../glossary/README.md#torchairtorch-图模式)。在推理开发过程中可能会遇到精度问题。

大模型精度调试工具（Large Language Model Debug Tool) 用于帮助开发者快速定位推理开发过程中精度问题，发现根因，提升开发效率。

## 大模型精度调试步骤

大模型精度调试定位，一般思路是从整网到算子，从外到内，从粗到细逐步定位根因，具体定位操作可以视情况调整。一般分为以下 3 个步骤：

1. 定位存在精度问题的输入
   1. **数据集评估精度定位**: 如果数据集评估不理想，可以通过工具定位数据集中存在精度问题的数据输入 (todo： 7.0.0 RC2 实现)
   2. **输出 token 精度定位**: 如果生成任务发现输出误差逐渐变大，可以通过工具定位首先出现精度问题的输出 token (todo： 7.0.0 RC2 实现)
2. 定位存在精度问题的算子
   1. **整网算子精度比对**: 当相同输入但是 npu 和 cpu(gpu)输出不一致，可以通过逐层算子比对方式定位到存在精度问题的算子。
      - 排查流程说明文档：
        - [**《整网算子精度比对-加速库推理场景》**](./整网算子精度比对-加速库推理场景.md)，加速库推理场景，如何定位存在精度问题算子
        - [**《整网算子精度比对-torchair 推理场景》**](./整网算子精度比对-torchair推理场景.md)，torchair 推理场景，如何定位存在精度问题算子
      - 相关功能：
        - [**《加速库 DUMP 功能使用说明》**](./加速库DUMP功能使用说明.md)，提供了 dump 加速库的网络结构、算子信息、推理输入输出等信息，支撑后续手动和自动比对、分析工作。
        - [**《自动比对功能使用说明》**](./自动比对功能使用说明.md)，提供了自动比对功能，比对标杆数据和推理数据之间的误差。
   2. **异常检测**: 定位推理过程中是否存在算子预算溢出、内存踩踏。可以参考[**《异常检测功能使用说明》**](异常检测功能使用说明.md)
3. 单算子分析
   1. **单算子精度预检**: 工具提供加速库内置算子的精度预检能力，根据模型推理时 dump 的 tensor 及算子信息，计算标杆 output，比较 dump 的算子 output 与标杆数据的误差，以检测算子精度是否达标。可以参考[**《精度预检功能使用说明》**](精度预检功能使用说明.md)

## 精度问题工具定位思路

精度问题一般是通过比对标杆数据和当前推理数据的误差来定位。标杆数据一般来自于 cpu 或者 gpu 推理结果。所以我们通常定位步骤如下：

1. dump：获取标杆数据和当前推理工程数据
   - 标杆数据 dump：数据来源一般来自于 cpu 或者 gpu 在线推理，通过当前工具提供的接口进行 dump
   - 当前推理工程数据 dump：根据推理的框架不同，通过不同方式进行数据 dump
   - dump 的数据类型有：整网输入输出、算子输入输出、kernel 输入输出、网络结构、等等
2. 精度比对：比对标杆数据和自己的数据
   - 如果比对算子输入输出，需要根据网络结构分析，进行算子映射。也可以手动写映射关系
   - 目前支持比对算法有：余弦相似度等，也支持自定义比对算法
   - 可以指定告警阈值
