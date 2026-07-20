# scratch-v2 第 3 折验收记录

- fold 3、seed 42：macro_f1 = `0.9152607256694294`。
- fold 3、seed 43：macro_f1 = `0.9060617492431067`。
- fold 3、seed 44：macro_f1 = `0.9052255617697261`。
- 本折三种子平均 macro_f1：`0.9088493455607540`。
- 三个单元均与旧 scratch 诊断值逐项完全一致，绝对差为 0。
- 新 scratch-v2 的 content-addressed patch provenance 均通过在线检查；旧 scratch 文件未改写。

结论：第 3 折通过预声明在线停止策略，可继续核验第 4 折。本记录不支持板级、功耗或 SURE 结论。
