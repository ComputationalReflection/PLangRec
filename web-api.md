# *PLangRec*

[![License](https://img.shields.io/github/license/computationalreflection/plangrec)](LICENSE) 
[![Latest release](https://img.shields.io/github/v/release/computationalreflection/plangrec?include_prereleases)](https://github.com/ComputationalReflection/PLangRec/releases)
<img alt="Code size" src="https://img.shields.io/github/languages/code-size/ComputationalReflection/PLangRec">
<img alt="Repo size" src="https://img.shields.io/github/repo-size/ComputationalReflection/PLangRec">

*PLangRec* is a system designed to recognize the programming language of a source code excerpt. 

*PLangRec* is provided as a Python desktop application, web API and web application.
In this document, we describe the desktop web API.

## Installation

First, install the latest release of Python 3. We have used Python 3.11, so any higher version should work. 

Second, install all the required packages, either using a virtual environment or modifying the system Python.
The simplest approach to install the required packages is to use the `web-api-requirements.txt` file with pip:

``` bash
pip install -r web-api-requirements.txt --upgrade
``` 

Third, place both `common` and the `web-api` folders as sibling directories. 
That is, they must be subdirectories of the same parent directory.

Finally, run the *PLangRec* web API as a Python program (it uses the Flask framework). 
The model and meta-model will be downloaded from the Internet, 
so the first execution may take minutes (be patient). 
This is how you run the web API:

``` bash
cd web-api
python main.py
``` 

## Functionality

The *PLangRec* web API offers a single 'predict' `GET` method.
It accepts the input source code via the 'source_code' parameter.
It returns a JSON document with the probabilities of the code being written in 21 
different programming languages.


## Example

Let's assume the web API is running in https://www.reflection.uniovi.es/plangrec/webapi/BRNN/

Since the web API only provides a `predict` function through the `GET` HTTP method,
we can use any web browser to test it 
([curl](https://curl.se/) is a widespread application widely used to test web APIs). 
If we write the following URL [https://www.reflection.uniovi.es/plangrec/webapi/BRNN/predict?source_code=select%20*%20from%20Customers;](https://www.reflection.uniovi.es/plangrec/webapi/BRNN/predict?source_code=select%20*%20from%20Customers;),
we are asking the web API to predict the programming language of code `select * from Customers;`.

This is the JSON response we would get:

``` json
{   "Assembly":0.0,
    "C":0.0,
    "C#":0.0,
    "C++":0.0,
    "CSS":0.0,
    "Go":0.0,
    "HTML":0.0,
    "Java":0.0,
    "JavaScript":0.0,
    "Kotlin":0.01,
    "Matlab":0.0,
    "PHP":0.0,
    "Perl":0.0,
    "Python":0.0,
    "R":0.0,
    "Ruby":0.01,
    "SQL":99.34,
    "Scala":0.01,
    "Swift":0.0,
    "TypeScript":0.0,
    "Unix Shell":0.62
}
```

You can see how SQL is the language with the highest probability (99.34%).


With this API, you can build your own applications that need to recognize the programming 
language of a source code excerpt. 

## License

[MIT license](LICENSE).
