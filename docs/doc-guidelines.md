# 文档编写规范

该文档规范 AIT 仓库中文档写作格式，行文逻辑，组织结构等。持续更新，大家有好的意见可以随时讨论~

# AIT 资料目录结构

- 根目录
  - README.md AIT 文档主页，介绍推理业务，以及 AIT 工具在推理业务中的作用
  - start-page.md 文档的开始页面，相当与封面页，是用户的第一印象
  - menu.md 菜单文档，是整个 AIT 文档的骨架，在界面中会体现在侧边栏和导航栏中
  - index.html 界面主页，负责界面展示。不需要变更
  - doc-guidelines.md 当前文档，文档编写规范
  - /install 安装相关文档
  - /assert 界面展示所需的 js 和 css 文件，不需要变更
  - /pics 存放 md 文档需要用到的图片
  - 其他目录为不同工具的说明文档
    - README.md 工具主页
    - xxx.md 具体场景，功能说明文档
    - FAQ.md FAQ
    - /history 历史版本
    - /pics 存放 md 文档需要用到的图片

# 行文逻辑

1. 一个独立工具，需要在该工具目录的下，添加一个 README.md 文档，作为工具主页，内容为工具总体介绍。包括以下：
   1. 功能介绍：通过场景引出功能的方式，更加易懂。工具一般是对应一个较大场景。比如：llm 对应加速库调优；benchmark 对应 推理运行以及测评。大的场景可能还包含若干个子场景。我们需要将场景讲清楚
      1. 场景的上下游；用户工作流程是怎么样的；
      2. 在用户工作流程中，可能会遇到哪些问题；
      3. 我们的有哪些功能可以在解决问题。
      4. 如果工具只包含单独一个功能，场景较简单，可以将功能使用直接在当前文档中说明清楚；如果是较复杂，涉及多个功能，多个场景，可以在单独的 md 文档中进行功能或者场景介绍。并在当前文档提供快速跳转链接（比如一个精度比对场景，包含 dump 和比对，可以单独一个文档进行描述。或者一个较独立的功能，可以单独一个文档进行描述）
   2. 工具后续发展计划
   3. 相关链接（API 列表，其他相关工具，安装指南，FAQ, 历史版本等）
   4. FAQ ，如果 FAQ 较多，可以单独创建一个 FAQ.md 文档。并提供跳转链接
   5. 如果工具需要保留历史版本资料，可以添加文件夹 history ，将历史文档放到 history 目录进行归档。并在工具主页添加对应链接
2. 工具中一个具体的场景或者功能介绍。添加一个 md 文档。包括一下内容：
   1. 具体的场景描述,一般先介绍主场景，主流程。特殊的在靠后章节进行补充介绍
   2. 工具的使用步骤
      1. 前置准备步骤，注意事项
      2. 正式工具使用步骤，建议介绍工具内部流程，方便问题排查和用户理解
      3. 工具结果如何查看与分析
      4. 异常说明
   3. 一些分支场景的描述，包括特殊场景，异常场景应该如何处理
   4. 不直接提供 API 参数列表，或者命令行参数列表。在单独的文档中提供。一般命名为 api.md 或者 cmd.md。可以在使用步骤中，包含跳转链接

# 写作格式

1. API 或命令参数

   1. 需要增加一列说明支持的版本（历史可以不填，后面新增的参数需要填写）
   2. 如果参数较多，需要增加一列进行参数分组
   3. 参数前后顺序尽量与--help 中一致

2. 其他待补充...