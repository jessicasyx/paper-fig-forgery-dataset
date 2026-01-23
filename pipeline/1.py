from pipeline.generators.repaint_wrapper import run_repaint_batch

PROJECT_ROOT = r"E:\GraduationProject\paper-fig-forgery-dataset"
REPAINT_ROOT = r"E:\GraduationProject\paper-fig-forgery-dataset\external\RePaint-main"
BASE_CONF = r"E:\GraduationProject\paper-fig-forgery-dataset\external\RePaint-main\confs\test_inet256_thick.yml"

ok, msg = run_repaint_batch(
    repaint_root=REPAINT_ROOT,
    base_conf_path=BASE_CONF,
    project_root=PROJECT_ROOT,
    timestep_respacing="250",
    batch_size=2,
    max_len=0,
)
print(ok, msg)
