# sp500-prediction
Trying to predict the S&P 500 using machine learning and natural language processing


# Set Up

## Dependencies

We are using conda to manage our dependencies. To install conda, follow the instructions [here](https://docs.conda.io/en/latest/)

The dependencies are listed in the `environment.yml` file. To install them, `cd` into the root directory and run the following command:

```bash
conda env create -f environment.yml
```

This will create a conda environment called `sp500-prediction-env`. To activate the environment, run the following command:

```bash
conda activate sp500-prediction-env
```

## Credentials

All API keys and other credentials are stored in a dotenv file. To set up your own credentials, create a file called `.env` in the root directory and add the API keys like so:

```bash
ALPHAVANTAGE_API_KEY=EX4MPLEAP1K3Y
NASDAQ_DATA_LINK_API_KEY=EX4MPLEAP1K3Y
NYT_API_KEY=EX4MPLEAP1K3Y
```

These will be loaded as environment variables when you run the code.

# Running the Code

The code is set up as a python package with modules. All code should be run from the root directory, and all imports should be relative to the root directory. To run a module, use the `-m` flag (more about this [here](https://docs.python.org/3/using/cmdline.html#cmdoption-m) and [here](https://stackoverflow.com/questions/50821312/what-is-the-effect-of-using-python-m-pip-instead-of-just-pip#:~:text=More%20general%20comments%20about%20the%20%2Dm%2Dflag%20(Dec.%202022))).

I recomend you configure your IDE to run the code this way instead of running the files directly from the command line.

