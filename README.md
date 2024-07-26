# *PLangRec*

[![License](https://img.shields.io/github/license/computationalreflection/plangrec)](LICENSE) 
[![Latest release](https://img.shields.io/github/v/release/computationalreflection/plangrec?include_prereleases)](https://github.com/ComputationalReflection/PLangRec/releases)
<img alt="Code size" src="https://img.shields.io/github/languages/code-size/ComputationalReflection/PLangRec">
<img alt="Repo size" src="https://img.shields.io/github/repo-size/ComputationalReflection/PLangRec">

*PLangRec* is a system designed to recognize the programming language of a source code excerpt. 
It utilizes a character-level deep learning model to predict the language of individual code lines, 
combined with a stacking ensemble meta-model that leverages the single-line predictions to identify 
the language from multiple lines of code. 
Our evaluation shows that *PLangRec* outperforms state-of-the-art systems for language recognition.


*PLangRec* is provided as a Python desktop application, web API and web application.

## Python desktop application

Make sure you have Python installed. Then, install all the required packages:

``` bash
pip install -r requirements.txt --upgrade
``` 

Download both the both `common` and the `desktop` folders. They must be
subdirectories of the same parent directory (the must be sibling directories).

Finally, run the *PLangRec* desktop application as a Python program. 
The model and meta-model will be downloaded from the Internet, 
so the first execution may take minutes (be patient):

``` bash
cd desktop-app
python main.py
``` 

![Desktop application screenshot](img/desktop-app.png)

For a more detailed description of the desktop Python application, please read [desktop application](desktop.md).


## Web API

Make sure you have Python installed. Then, install all the required packages:

``` bash
pip install -r requirements.txt --upgrade
``` 

Download both the both `common` and the `web-api` folders. 
They must be subdirectories of the same parent directory (the must be sibling directories).

Finally, run *PLangRec* as a Web API. 
The model and meta-model and will be downloaded from the Internet, 
so the first execution may take minutes:

``` bash
cd web-api
python main.py
``` 
Please, see [web API](web-api.md) for more details about *PLangRec*'s web API.


## Web application

For the web application, you need to deploy the Web API first
because the application consumes the API.

Steps:

1. Deploy the Web API in one server. 
2. Download the `web-app` folder to your web server. 
3. Modify the value of the `WEB_SERVER` variable in the `index.html` file, 
setting its value to the server where you have the web API (step 1).    

The Web application will be ready to work, calling the Web API.

![Web application screenshot](img/web-app.png)

For a more detailed description of the web application, please read [web application](web-app.md).

## Models

*PLangRec* uses a single-line deep model classifier to predict the programming language from the source code.
It uses a bidirectional recurrent neural network (BRNN). We have also tried a multi-layer preceptron (MLP)
architecture, but the BRNN achieved better performance. 
The `BRNN` and `MLP` directories included in this repository include the training, validation and evaluation
of both models.

When multiple lines of code are available, *PLangRec* uses a stacking ensemble meta-model to predict 
the programming language. It is a MLP artificial neural network that combines the single-line predictions
to identify the language from multiple lines of code. The `meta-model` directory included in this repository
includes the training, validation and evaluation of the meta-model.

## License

[MIT license](LICENSE).
