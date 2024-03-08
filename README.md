# *PLangRec*

[![License](https://img.shields.io/github/license/ComputationalReflection/plangrec)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/computationalreflection/plangrec?include_prereleases)](https://github.com/ComputationalReflection/PLangRec/releases)
<img alt="Code size" src="https://img.shields.io/github/languages/code-size/ComputationalReflection/PLangRec">
<img alt="Repo size" src="https://img.shields.io/github/repo-size/ComputationalReflection/PLangRec">

*Programming language recognizer.*

*PLangRec* is a character-level deep model to recognize the programming language of source code.

*PLangRec* is provided as a Python desktop application, web API and web application.

## Python desktop application

Make sure you have Python installed. Then, install all the required packages:

``` bash
    pip install -r requirements.txt --upgrade
``` 

Make sure you have downloaded both the both `common` and the `desktop` folders, and they are
both subdirectories of the same parent directory (the must be sibling directories).

Finally, run *PLangRec* as a Python application (the model will be downloaded from the Internet, 
so the first execution may take minutes):

``` bash
    cd desktop-app
    python main.py
``` 

![Desktop application screenshot](img/desktop-app.png)

## Web API

Make sure you have Python installed. Then, install all the required packages:

``` bash
    pip install -r requirements.txt --upgrade
``` 

Make sure you have downloaded both the both `common` and the `web-api` folders, and they are
both subdirectories of the same parent directory (the must be sibling directories).

Finally, run *PLangRec* as a Web API (the model will be downloaded from the Internet, 
so the first execution may take minutes):

``` bash
    cd web-api
    python main.py
``` 

## Web application

For the web application, you need to deploy the Web API
because the application consumes the API.

Steps:

1. Deploy the Web API in one server. 
2. Download the `web-app` folder to your web server. 
3. Modify the value of the `WEB_SERVER` variable in the `index.html` file, 
setting its value to the server where you have the web API (step 1).    

The Web application will be ready, calling the Web API.

![Web application screenshot](img/web-app.png)

## License

[MIT license](LICENSE).
