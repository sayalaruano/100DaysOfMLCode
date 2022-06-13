import pandas as pd
import pickle

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.flow_runners import SubprocessFlowRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger("Init logger")
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger("Init logger")

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger("Init logger")

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    logger = get_run_logger("logger")
    orig_path = "../../Data/fhv_tripdata"
    date_format = "%Y-%m"

    if date is None:
        today = datetime.today()
        train_date = (today-relativedelta(months=2)).strftime(date_format)
        val_date = (today-relativedelta(months=1)).strftime(date_format)
    else:
        new_date = datetime.strptime(date, "%Y-%m-%d")
        train_date = (new_date-relativedelta(months=2)).strftime(date_format)
        val_date = (new_date-relativedelta(months=1)).strftime(date_format)

    train_path = f"{orig_path}_{train_date}.parquet"
    val_path = f"{orig_path}_{val_date}.parquet"

    logger.info(f"Train path: {train_path}")
    logger.info(f"Val path: {val_path}")
    return train_path, val_path

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    train_path, val_path = get_paths(date).result()
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # save preprocessor
    with open(f'artifacts/dv-{date}.b', 'wb') as f_out:
        pickle.dump(dv, f_out)

    # save model
    with open(f'artifacts/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

#main(date="2021-08-15")

DeploymentSpec(
    flow=main,
    name="first_deployment",
    schedule=CronSchedule(cron="0 9 15 * *"),
    tags=['homework_week3'],
    flow_runner=SubprocessFlowRunner(),
)
