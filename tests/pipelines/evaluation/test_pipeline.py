from src.gamereviewratio.pipelines.evaluation import create_pipeline


# testuje czy pipeline evaluation buduje się poprawnie
def test_pipeline_builds_successfully():
    pipeline = create_pipeline()

    assert pipeline is not None, "pipeline nie został utworzony"
    assert len(pipeline.nodes) > 0, "pipeline nie zawiera żadnych node'ów"
