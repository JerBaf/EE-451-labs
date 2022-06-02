from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
import selenium
import time
import pathlib
import json
import re
import numpy as np
from joblib import Parallel, delayed

def setup_browser(headless):
    # Setup browser
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("headless")
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": str(pathlib.Path().resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False
    })
    chrome_options.add_argument("--log-level=3")
    driver = webdriver.Chrome(
        ChromeDriverManager().install(), options=chrome_options)
    params = {'behavior': 'allow',
              'downloadPath': str(pathlib.Path().resolve())}
    driver.execute_cdp_cmd('Page.setDownloadBehavior', params)
    return driver


def format_annotations(text):
    s = text
    s = s.replace("\n","")
    s = s.replace(" ","")
    s = s.replace("\"","")
    l = s.split(",")
    l = list(map(lambda s: s.split("{")[-1],l))
    l = list(map(lambda s: s.split("}")[0],l))
    predictions = {}
    for i in range(len(l)//6):
        pred = l[6*i:6*(i+1)]
        card = pred[4]
        if card not in predictions:
            predictions[card] = pred
    return predictions

def retrieve_annotation(image_path,hidden=True):
    driver = setup_browser(hidden)
    driver.delete_all_cookies()
    time.sleep(5)
    startup_page = "https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/1/try"
    driver.get(startup_page)
    ### Interact with browser
    sign_in_waiting = True
    time.sleep(1)
    while sign_in_waiting:
        try:
            sign_in  = driver.find_element(By.CSS_SELECTOR,"#content > div.navbarContainer > nav > div:nth-child(5) > div > a.signIn.signInButton")
            sign_in_waiting = False
        except:
            time.sleep(1)
    sign_in.click()
    time.sleep(1)
    mail = driver.find_element(By.CSS_SELECTOR,"#message > div > div.firebaseui-card-content > form > ul > li:nth-child(1) > button")
    mail.click()
    time.sleep(1.2)
    mail_enter = driver.find_element(By.CSS_SELECTOR,"#message > div > form > div.firebaseui-card-content > div > div.firebaseui-textfield.mdl-textfield.mdl-js-textfield.mdl-textfield--floating-label.is-upgraded > input")
    mail_enter.send_keys("iapr2022project@gmail.com")
    time.sleep(1.1)
    mail_validation = driver.find_element(By.CSS_SELECTOR,"#message > div > form > div.firebaseui-card-actions > div > button.firebaseui-id-submit.firebaseui-button.mdl-button.mdl-js-button.mdl-button--raised.mdl-button--colored")
    mail_validation.click()
    time.sleep(1)
    password = driver.find_element(By.CSS_SELECTOR,"#message > div > form > div.firebaseui-card-content > div:nth-child(3) > input")
    password.send_keys("ProjectAPI")
    time.sleep(1.1)
    submit_sign_in = driver.find_element(By.CSS_SELECTOR,"#message > div > form > div.firebaseui-card-actions > div.firebaseui-form-actions > button")
    submit_sign_in.click()
    time.sleep(3)
    ### Submit image
    anot = driver.find_element(By.CSS_SELECTOR,"#fileInput")
    anot.send_keys(image_path)
    time.sleep(3)
    waiting = True
    while waiting:
        try:
            driver.delete_all_cookies()
            time.sleep(1)
            wrong = driver.find_element(By.CSS_SELECTOR,"#swal2-title")
            wrong_button = driver.find_element(By.CSS_SELECTOR,"body > div.swal2-container.swal2-center.createDatasetDialog.swal2-backdrop-show > div > div.swal2-actions.rightActions > button.swal2-confirm.actionButton.primary")
            wrong_button.click()
            time.sleep(1)
            driver.refresh()
            time.sleep(5)
            anot = driver.find_element(By.CSS_SELECTOR,"#fileInput")
            anot.send_keys(image_path)
            time.sleep(2)
        except:
            waiting = False
    annotations =  driver.find_element(By.CSS_SELECTOR,"#inferenceJson")
    ### Format Annotations
    annotations_list = format_annotations(annotations.text)
    driver.close()
    return annotations_list