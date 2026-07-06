from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    use_case: str
    url: str
    note: str
    stage: str


DATASETS = (
    DatasetInfo(
        name="Mozilla Common Voice Yue",
        use_case="Primary Cantonese ASR baseline",
        url="https://commonvoice.mozilla.org/zh-HK/datasets",
        note="Use local cv-corpus-26.0-2026-06-12/yue train/dev/test first.",
        stage="primary",
    ),
    DatasetInfo(
        name="MDCC",
        use_case="Future Hong Kong Cantonese ASR research",
        url="https://arxiv.org/abs/2201.02419",
        note="Paper describes a 73.6-hour multi-domain Cantonese corpus.",
        stage="future",
    ),
    DatasetInfo(
        name="CantoMap",
        use_case="Future controlled task-oriented Cantonese speech",
        url="",
        note="Candidate source; verify public access and license before use.",
        stage="future",
    ),
    DatasetInfo(
        name="HKCanCor",
        use_case="Future linguistic analysis and transcript normalization",
        url="https://pycantonese.org/",
        note="Treat as a linguistic/text resource unless audio access is confirmed.",
        stage="future",
    ),
    DatasetInfo(
        name="WenetSpeech-Yue",
        use_case="Future large-scale Cantonese ASR research",
        url="https://arxiv.org/abs/2509.03959",
        note="Paper describes 21,800 hours; verify release status and license.",
        stage="future",
    ),
)


def dataset_names() -> list[str]:
    return [dataset.name for dataset in DATASETS]
