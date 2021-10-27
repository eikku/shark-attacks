from valohai import Pipeline

def main(config) -> Pipeline:

    #Create a pipeline called "mypipeline".
    pipe = Pipeline(name="sharkpipe", config=config)

    # Define the pipeline nodes.
    fetch = pipe.execution("fetch_data")
    process = pipe.execution("pre_process")
    pepare_text = pipe.execution("pepare_text")
    fine_tune = pipe.execution("fine_tune")

    # Configure the pipeline, i.e. define the edges.
    fetch.output("*").to(process.input("attacksmini"))
    process.output("*").to(pepare_text.input("attacksminiprocessed"))
    pepare_text.output("train.csv").to(fine_tune.input("train"))
    pepare_text.output("val.csv").to(fine_tune.input("val"))
    pepare_text.output("test.csv").to(fine_tune.input("test"))
    pepare_text.output("my_dict.csv").to(fine_tune.input("my_dict"))

    return pipe
