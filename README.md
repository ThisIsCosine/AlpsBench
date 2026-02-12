# AlpsBench: A Benchmark for Realistic Personalized Memory Systems

## 🚧 Dataset Availability: Coming Soon

**The full AlpsBench dataset is currently under internal review.**

Due to the **GitHub platform's strict file size limitations and storage quotas** regarding large-scale benchmarks, we are currently encountering upload restrictions that prevent hosting the full dataset directly. We are actively resolving these platform-specific technical issues and setting up an alternative hosting solution to ensure stable access. We expect to release the download link in the near future.

Thank you for your patience and understanding.


## � Task Examples & Directory Structure

While the full data is being prepared, we provide illustrative samples for each core task in the `examples/` directory.

### Folder Structure

The `examples` folder is organized by task type:

```text
examples/
├── task1/
│   └── task1_dataset.json
├── task2/
│   └── task2_dataset.json
├── task3/
│   └── task3_dataset_d100.json
└── task4/
    ├── ability1.json
    ├── ability2.json
    ├── ability3.json
    ├── ability4.json
    └── ability5.json
```

### File Descriptions

Below is a brief explanation of what each sample file contains:

#### Task 1: Implicit Memory Extraction
*   **`examples/task1/task1_dataset.json`**
    Demonstrates **Personalized Information Extraction**: transforming raw dialogue history into structured memory entries. Each entry captures attributes such as `Memory ID`, `Type` (Direct/Indirect), `Label`, `Value`, `Confidence`, and supporting evidence, reflecting the assistant's understanding of user preferences and background.

#### Task 2: Memory Update & Conflict Resolution
*   **`examples/task2/task2_dataset.json`**
    Demonstrates **Personalized Information Update**: handling the dynamic evolution of user preferences. Given an existing memory set and new dialogue history, the system must determine the correct action—*Retention* (filtering noise), *Addition* (new preferences), or *Modification* (resolving conflicts/updating outdated info)—and output the updated memory state.

#### Task 3: Evidence-based Memory Retrieval
*   **`examples/task3/task3_dataset_d100.json`**
    Demonstrates **Personalized Information Retrieval**: identifying relevant insights from a memory pool to answer a user query ($Q$). The task is to retrieve the correct positive memory ($M_{pos}$) from a candidate set containing one positive sample mixed with random negative samples ($M_{neg}$). 
    *Note: This example shows the setting with **100 distractors**. The full benchmark includes larger pools (300, 500, 700, 1000 distractors) to test retrieval robustness.*

#### Task 4: End-to-End Personalized Generation
Demonstrates **Personalized Information Utilization**: generating an answer ($\hat{A}$) to a user query ($Q$) based on dialogue history ($H$). The evaluation covers five core dimensions, with each example file corresponding to a specific ability:

*   **`examples/task4/ability1.json` (Persona Awareness)**:  
    Assesses if the assistant correctly recalls and integrates explicit user attributes (e.g., occupation, education) into responses.
*   **`examples/task4/ability2.json` (Preference Following)**:  
    Measures the ability to infer latent, dynamic preferences from history (e.g., specific tastes for recommendations) rather than offering generic advice.
*   **`examples/task4/ability3.json` (Virtual–Reality Awareness)**:  
    Evaluates the distinction between real user information and role-playing/fictional content to prevent "in-character" data from contaminating real-world assistance.
*   **`examples/task4/ability4.json` (Constraint Following)**:  
    Examines whether the assistant respects previously stated negative constraints or specific exclusions .
*   **`examples/task4/ability5.json` (Emotional Intelligence)**:  
    Assesses the capacity to provide differentiated emotional responses  based on the user's historical emotional state and needs.
