from valohai import Pipeline

def main(config) -> Pipeline:

    #Create a pipeline called "mypipeline".
    pipe = Pipeline(name="sharkpipe", config=config)

    # Define the pipeline nodes.
    fetch = pipe.execution("fetch_data")
    process = pipe.execution("pre_process")
    tokenize_text = pipe.execution("tokenize_text")
    fine_tune = pipe.execution("fine_tune")

    # Configure the pipeline, i.e. define the edges.
    fetch.output("*").to(process.input("attacksmini"))
    process.output("*").to(tokenize_text.input("attacksminiprocessed"))
    tokenize_text.output("*").to(fine_tune.input("train_dataset"))
    tokenize_text.output("*").to(fine_tune.input("val_dataset"))
    tokenize_text.output("*").to(fine_tune.input("test_dataset"))

    return pipe
