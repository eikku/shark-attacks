from valohai import Pipeline

def main(config) -> Pipeline:

    #Create a pipeline called "mypipeline".
    pipe = Pipeline(name="mypipeline", config=config)

    # Define the pipeline nodes.
    fetch = pipe.execution("fetch_data")
    process = pipe.execution("pre_process")

    # Configure the pipeline, i.e. define the edges.
    fetch.output("*").to(process.input("attacksmini"))

    return pipe
