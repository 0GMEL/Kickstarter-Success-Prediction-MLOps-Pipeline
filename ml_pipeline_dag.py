from airflow.sdk import dag
from airflow.sdk import task
from datetime import datetime, timedelta

WORKING_DIR = '/Users/og_mel/PycharmProjects/kst'

default_args = {
    'owner': 'og_mel',
    'depends_on_past': False,
    'start_date': datetime(1997, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
    'ml_pipeline',
    default_args=default_args,
    description='first dag for ml',
    schedule=None,
    catchup=False,
    tags=['ml', 'dvc']
)

def ml_pipeline():
    @task.bash
    def check_environment():
        return f'cd {WORKING_DIR} && export PATH=$PATH:~/.local/bin && echo "success"'

    # @task.bash
    # def init_dvc():
        # return f'cd {WORKING_DIR} && export PATH=$PATH:~/.local/bin && dvc init && dvc status'

    # @task.bash
    # def pull_data():
        # return f'cd {WORKING_DIR} && export PATH=$PATH:~/.local/bin && dvc pull'

    @task.bash
    def run_dataloader():
        return f'cd {WORKING_DIR} && python3 data_loader.py && ls -la *.csv'

    @task.bash
    def run_training():
        return f'cd {WORKING_DIR} && mkdir -p model_weights && python3 train_model.py && ls -la model_weights'

    @task.bash
    def run_dvc_repro():
        return f'cd {WORKING_DIR} && export PATH=$PATH:~/.local/bin && dvc repro --force'

    check_task = check_environment()
    # init_task = init_dvc()
    # pulldata_task = pull_data()
    dataloader_task = run_dataloader()
    training_task = run_training()
    repro_task = run_dvc_repro()

    check_task >> dataloader_task >> training_task >> repro_task


ml_pipeline_dag = ml_pipeline()