import os
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import (
    Reader,
    PSI,
    SSHELR,
    Evaluation
)

base_path = os.path.abspath(os.path.dirname(__file__))
guest_data_path = os.path.join(base_path, "breast_hetero_guest.csv")
host_data_path = os.path.join(base_path, "breast_hetero_host.csv")

data_pipeline = FateFlowPipeline().set_parties(local="0")
guest_meta = {
    "delimiter": ",", "dtype": "float64", "label_type": "int64",
    "label_name": "y", "match_id_name": "id"
}
host_meta = {
    "delimiter": ",", "input_format": "dense", "match_id_name": "id"
}
data_pipeline.transform_local_file_to_dataframe(
    file=guest_data_path,
    namespace="experiment", name="breast_hetero_guest",
    meta=guest_meta, head=True, extend_sid=True
)
data_pipeline.transform_local_file_to_dataframe(
    file=host_data_path,
    namespace="experiment", name="breast_hetero_host",
    meta=host_meta, head=True, extend_sid=True
)

pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")

reader_0 = Reader("reader_0")
reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")

psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

hetero_lr_0 = SSHELR(
    "hetero_lr_0",
    epochs=3,
    early_stop="diff",
    learning_rate=0.1,
    batch_size=64,
    init_param={"init_method": "random"},
    threshold=0.5,
    train_data=psi_0.outputs["output_data"],
    validate_data=psi_0.outputs["output_data"]
)

evaluation_0 = Evaluation(
    "evaluation_0",
    runtime_parties=dict(guest="9999"),
    metrics=["auc", "binary_accuracy"],
    input_datas=[hetero_lr_0.outputs["train_output_data"]]
)

pipeline.add_tasks([reader_0, psi_0, hetero_lr_0, evaluation_0])

pipeline.compile()
pipeline.fit()

print("Model Summary:")
print(pipeline.get_task_info("hetero_lr_0").get_output_model())
print("Evaluation Results:")
print(pipeline.get_task_info("evaluation_0").get_output_metric())

pipeline.dump_model("./pipeline_lr.pkl")

predict_pipeline = FateFlowPipeline()

pipeline = FateFlowPipeline.load_model("./pipeline_lr.pkl")

pipeline.deploy([pipeline.psi_0, pipeline.hetero_lr_0])

deployed_pipeline = pipeline.get_deployed_pipeline()
reader_1 = Reader("reader_1")
reader_1.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_1.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]

predict_pipeline.add_tasks([reader_1, deployed_pipeline])

predict_pipeline.compile()
predict_pipeline.predict()