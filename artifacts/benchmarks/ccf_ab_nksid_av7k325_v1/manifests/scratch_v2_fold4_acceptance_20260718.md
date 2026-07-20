# scratch-v2 第 4 折验收记录

- fold 4、seed 42：macro_f1 = `0.8948801178259115`。
- fold 4、seed 43：macro_f1 = `0.9095418098566077`。
- fold 4、seed 44：macro_f1 = `0.9352478911994927`。
- 本折三种子平均 macro_f1：`0.9132232729606706`。
- 三个单元均与旧 scratch 诊断值逐项完全一致，绝对差为 0。
- 新 scratch-v2 的 content-addressed patch provenance 均通过在线检查；旧 scratch 文件未改写。

结论：第 4 折通过预声明在线停止策略，scratch-v2 已形成完整 15 单元结果。最终可声明性仍以独立审计和测量优先总账为准。
