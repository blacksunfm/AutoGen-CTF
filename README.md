运行在python3.10环境

## 基础库安装

执行以下代码安装autogen库

```shell
pip install --upgrade pyautogen
```

## 环境变量

使用下面的命令设置环境变量(bash)<br/>
其中<...>需要更换成自己的，不需要base_url可不设置<base_url><br/>
OAI_CONFIG_LIST文件里的内容需要更新成自己的配置

```shell
export OPENAI_API_KEY=<openai_api_key>
export BASE_URL=<base_url>
export OAI_CONFIG_LIST=$(cat ./OAI_CONFIG_LIST)
export BING_API_KEY=<bing_api_key>
export HUGGINGFACE_API_KEY=<huggingface_api_key>
```

## 单个agent测试

实际操作浏览器的agent

```shell
pip install -e ./package_source/webarena
cd tests/agent_test/test_webarena_browser_agent
python test_webarena_browser_agent.py
```

其他agent类似<br/>
(由于版本更新，rag_assistant似乎不能用了，需要重构一下)

## benchmark测试

安装相关库

```shell
pip install autogenbench
```

以CTFAIA为例

```shell
cd tests/benchmark_test/CTFAIA
```

### Running the init_tasks.py
```shell
python Scripts/init_tasks.py
```

### Running the TwoAgents tasks

Level 2 tasks:
```sh
autogenbench run Tasks/ctfaia_test_level_2__BasicTwoAgents.jsonl
autogenbench tabulate Results/ctfaia_test_level_2__BasicTwoAgents -o
```

Level 1 and 3 tasks are executed similarly.

the result.json can be submitted to https://huggingface.co/spaces/autogenCTF/agent_ctf_leaderboard






