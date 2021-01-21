import pytest
import ipynb
import pandas as pd
import numpy as np
from ipynb.fs.full.linear_regression import create_df , train_model, rmse

pytest.df = create_df("testdf.csv")
pytest.x = pytest.df[['total_bill','sex',"smoker","size","Sat", "Sun", "Thur", "Lunch"]]
pytest.y = pytest.df["tip"]

def test_sanity_check1():
    assert pytest.df.iloc[0]["Sat"] == 0

def test_sanity_check2():    
    assert pytest.df.iloc[0]["Sun"] == 1

def test_sanity_check3():
    assert pytest.df.iloc[25]["sex"] == 1

def test_sanity_check4():
    assert pytest.df.iloc[25]["Thur"] == 1
    
def test_rmse1():
    rmse_1 = rmse(np.array([10,20,30,40]), np.array([11,12,13,14]))
    assert float("{0:.2f}".format(rmse_1))== 16.05

def test_rmse2():
    rmse_2 = rmse(np.array([1,3,5]), np.array([2,4,8]))
    assert float("{0:.2f}".format(rmse_2)) == 1.91

def test_model():
    model = train_model(pytest.x, pytest.y)
    pred = model.predict(pytest.df[['total_bill','sex',"smoker","size","Sat", "Sun", "Thur", "Lunch"]])
    error_1 = rmse(pytest.df["tip"][0:20], pred[0:20])
    error_2 = rmse(pytest.df["tip"][20:30], pred[20:30])
    assert float("{0:.2f}".format(error_1)) == 1.30
    assert float("{0:.2f}".format(error_2)) == 0.69