# Cause-Aware Model Degradation Detection in AI-Native RAN

### An Intelligent Multi-Agent Framework for Monitoring, Drift Detection, Conflict Identification, and Root-Cause Analysis

## 📌 Overview

AI-Native Radio Access Networks (RANs) increasingly rely on intelligent applications, known as **xApps**, to optimize network performance and automate control decisions. However, as network conditions evolve and multiple xApps operate simultaneously, AI models may experience performance degradation due to changing data distributions, evolving relationships between inputs and outputs, conflicting control actions, and cascading effects.

**Cause-Aware Model Degradation in AI-Native RAN** is an intelligent, multi-agent framework designed to monitor model performance, detect degradation, identify its underlying causes, and support appropriate corrective actions.

Unlike conventional monitoring systems that focus on detecting performance drops or individual drift events, this project aims to distinguish between multiple potential causes of degradation and combine evidence from specialized detection agents to support explainable, cause-aware decision-making.

The framework integrates a monitoring agent, data drift detection, concept drift detection, xApp conflict detection, root-cause analysis, and a decision-making pipeline for validating and managing corrective actions.

## 🎯 Project Objectives

- Continuously monitor AI model and network performance in an AI-Native RAN environment.
- Detect persistent performance degradation and trigger cause-specific analysis.
- Identify **data drift** caused by changes in the statistical distribution of input features.
- Detect **concept drift** caused by changes in the relationship between input features and target outcomes.
- Identify potential conflicts between interacting xApps.
- Analyze evidence from multiple detection agents to identify likely degradation causes.
- Investigate cascading effects in which one degradation event may contribute to further performance deterioration.
- Generate explainable diagnostic results and recommend appropriate corrective actions.
- Validate proposed interventions before controlled deployment.



### Architecture Components

**1. Near-RT RIC and Data Lake**

The Near-Real-Time RAN Intelligent Controller (Near-RT RIC) represents the network-control environment. Network measurements, model outputs, and relevant operational information are collected and made available through the data layer.

**2. Monitoring Agent**

The monitoring agent acts as the first stage of the framework. It observes relevant performance indicators and detects potential degradation. When degradation persists or crosses defined thresholds, it triggers the specialized cause-detection agents.

**3. Cause Detection Layer**

This layer consists of three specialized agents that independently investigate different potential causes of degradation.

**4. Attribute Engine**

The attribute engine consolidates the evidence produced by the specialized agents and extracts relevant diagnostic attributes to support cause identification.

**5. Decision Agent**

The decision agent evaluates the available evidence and determines an appropriate diagnostic or mitigation response.

**6. Cascading Effect Analysis**

This component investigates whether degradation in one component may contribute to further degradation elsewhere in the system.

**7. Validation and Controlled Deployment**

Proposed corrective actions are evaluated before deployment. Validated actions can then be applied in a controlled manner, followed by renewed monitoring to assess their effectiveness.

## 🔍 Core Features

### 1. Intelligent Monitoring Agent

The monitoring agent continuously evaluates the available monitoring signals to identify potential model or network performance degradation.

Key responsibilities:

- Observe relevant network and model performance indicators.
- Compare current behavior against configured thresholds or baseline conditions.
- Identify persistent degradation rather than relying solely on isolated anomalies.
- Trigger specialized cause-detection agents when further investigation is required.
- Support continued monitoring after corrective actions.

### 2. Data Drift Detection Agent

Data drift occurs when the statistical distribution of input data changes over time.

For example, radio conditions, traffic patterns, or channel-quality measurements may change, causing the model to receive inputs that differ from its reference data.

The Data Drift Agent examines changes in feature distributions using statistical techniques such as:

- **Kolmogorov–Smirnov (KS) test:** Detects differences between reference and current distributions.
- **Wasserstein distance:** Measures the distance between probability distributions.
- **Population Stability Index (PSI):** Quantifies changes in feature distributions across reference and current datasets.

The agent produces drift indicators and supporting evidence for the cause-analysis stage.

### 3. Concept Drift Detection Agent

Concept drift occurs when the relationship between input features and the target outcome changes over time.

A model may continue receiving statistically similar inputs while its predictions become less accurate because the underlying system behavior has changed.

The Concept Drift Agent focuses on changes in predictive performance and the relationship between inputs and outcomes.

Potential techniques include:

- Prediction-error and model-performance monitoring.
- **ADWIN (Adaptive Windowing):** Detects changes in a monitored data stream using an adaptive window.
- Comparison of model behavior across different time windows.

Where suitable labels or ground-truth outcomes are available, the agent uses prediction errors and performance changes to identify potential concept drift.

### 4. xApp Conflict Detection Agent

In an AI-Native RAN, multiple xApps may independently optimize different network objectives. Their control actions can interact and potentially interfere with one another.

For example, one xApp may adjust a network parameter to improve throughput, while another changes a related parameter to improve interference management. Their combined behavior may produce an undesirable outcome.

The xApp Conflict Agent investigates potential conflicts using relevant interaction and operational features.

The framework can incorporate:

- Interaction features derived from xApp behavior and network measurements.
- **XGBoost** for conflict classification when suitable training data are available.
- **SHAP (SHapley Additive exPlanations)** to help explain feature contributions to model predictions.

The objective is to identify potential xApp interference as a distinct cause of degradation rather than incorrectly attributing every performance drop to data or concept drift.

### 5. Evidence Aggregation and Root-Cause Analysis

Each specialized agent contributes evidence about its respective degradation mechanism.

The framework combines these results to support cause-aware diagnosis.

The analysis considers:

- Data distribution changes.
- Changes in model predictive behavior.
- Potential interactions and conflicts between xApps.
- Relationships between detected events and observed degradation.
- Possible cascading effects across components.

Multiple causes may coexist. Therefore, the framework should not assume that every degradation event has exactly one cause.

### 6. Cascading Degradation Analysis

A degradation event can trigger or amplify other problems in a complex network environment.

For example, a change in network conditions may reduce prediction accuracy, leading to suboptimal xApp decisions that subsequently affect other network performance indicators.

The framework investigates such possible chains of events to help distinguish an isolated issue from a wider cascading problem.

### 7. Decision and Mitigation Pipeline

After cause analysis, the decision agent uses the available evidence to determine a suitable response.

Potential responses include:

- Recommending model retraining or recalibration.
- Recommending updates to reference data or monitoring thresholds.
- Investigating conflicting xApp control policies.
- Prioritizing suspected upstream causes in a cascading degradation chain.
- Requesting further observation when evidence is insufficient.

Corrective actions should be validated before deployment. Automated changes to a live network require appropriate safeguards, authorization, and rollback mechanisms.

## 🤖 Multi-Agent Workflow

The framework separates monitoring, diagnosis, and decision-making into distinct responsibilities.

| Agent / Component   | Primary Responsibility                                       |
| ------------------- | ------------------------------------------------------------ |
| Monitoring Agent    | Detect performance degradation and trigger investigation     |
| Data Drift Agent    | Detect changes in input-feature distributions                |
| Concept Drift Agent | Identify changes in predictive relationships or performance  |
| xApp Conflict Agent | Identify potential conflicts between xApp interactions       |
| Attribute Engine    | Consolidate diagnostic attributes and supporting evidence    |
| Decision Agent      | Select or recommend the next diagnostic or corrective action |

The cause-specific agents can be orchestrated using a graph-based agent workflow, such as LangGraph, to manage execution, evidence collection, and transitions between stages.

## 🧪 Detection Techniques

| Technique                   | Purpose                                                         |
| --------------------------- | --------------------------------------------------------------- |
| KS Test                     | Compare reference and current feature distributions             |
| Wasserstein Distance        | Quantify distribution differences                               |
| PSI                         | Measure feature-distribution stability                          |
| ADWIN                       | Detect changes in monitored data streams                        |
| Prediction-Error Monitoring | Identify deterioration in predictive performance                |
| XGBoost                     | Classify potential xApp conflicts when trained on suitable data |
| SHAP                        | Explain feature contributions to conflict-model predictions     |

The techniques are complementary: data drift does not automatically imply concept drift, and neither necessarily proves that a model's performance has degraded. Likewise, identifying an xApp conflict requires appropriate interaction evidence.

## 🛠️ Technology Stack

The exact technologies depend on the implementation and configuration of the project.

- **Programming Language:** Python
- **Multi-Agent Orchestration:** LangGraph
- **Machine Learning:** XGBoost
- **Explainability:** SHAP
- **Statistical Analysis:** KS test, Wasserstein distance, and PSI
- **Concept Drift Monitoring:** ADWIN and prediction-error monitoring
- **Data Processing:** Python data-processing libraries
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

Additional libraries, data stores, dashboards, and deployment tools can be documented here as they are integrated into the implementation.

##

## ⚙️ Installation and Setup

### Prerequisites

- Python version compatible with the project's dependencies.
- Git.
- Visual Studio Code or another Python development environment.
- Access to the required datasets or network measurements.
- Valid configuration and credentials for any external database or service used.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd cause-aware-ran
```

Replace the placeholder with your actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the file is not yet available, create it using the dependencies required by the actual implementation. Avoid installing packages that the project does not use.

### 4. Configure the Environment

If the application uses a database or external services, configure the required settings through environment variables or an appropriate configuration file.

For example, a `.env.example` file may document settings such as:

```text
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
```

These variables are illustrative; use the names expected by the actual application.

Do not commit passwords, API keys, or production credentials to GitHub.

### 5. Run the Application

Run the actual application entry point, once identified in the repository. For example:

```bash
python main.py
```

The command above is only an example; replace it with the correct entry point for your project.

## 📊 Expected Outputs

Depending on the implemented modules and available data, the framework can produce:

- Monitoring status and degradation alerts.
- Feature-level data drift measurements.
- Concept drift indicators and predictive-performance trends.
- xApp conflict classifications and explanatory evidence.
- Consolidated cause-analysis results.
- Possible cascading-degradation relationships.
- Recommended corrective actions.
- Validation results following an intervention.

These outputs help developers and network operators understand not only whether degradation has occurred, but also which mechanisms may explain it and what should be investigated next.

## 🌐 Application Domain

The framework is designed for research and experimentation in AI-Native Radio Access Networks, particularly environments where multiple intelligent applications interact with changing network conditions.

Potential applications include:

- Network performance monitoring.
- AI model reliability assessment.
- Online drift detection.
- xApp interaction analysis.
- Explainable root-cause diagnosis.
- Controlled model and policy adaptation.

## 🚧 Limitations and Future Enhancements

Potential future enhancements include:

- Real-time integration with Near-RT RIC telemetry.
- More robust multi-cause diagnosis.
- Improved detection of cascading dependencies.
- Evaluation using realistic or simulated RAN datasets.
- Integration with live model-performance monitoring.
- Automated testing of corrective actions.
- Feedback loops for assessing mitigation effectiveness.
- Dashboard-based visualization of alerts, causes, and recommendations.
- Safe deployment and rollback of validated changes.

Actual performance depends on data quality, reference distributions, model assumptions, detection thresholds, and the availability of suitable ground-truth labels.

## 🔐 Responsible Deployment

The framework is intended to support diagnosis and controlled mitigation. Recommendations should be evaluated before changes are applied to live network systems.

Access to telemetry, databases, and network-control interfaces must follow the relevant authorization and security policies. Sensitive credentials and private datasets should not be included in the repository.

## 👥 Project Team

**Project:** Cause-Aware Model Degradation in AI-Native RAN

- Dhanam V.
- Risah Ruth
- Sarumathi S.
- Samyuktha B.

## 📌 Conclusion

Cause-Aware Model Degradation in AI-Native RAN proposes a multi-agent approach to monitoring AI model degradation and investigating its underlying causes.

By bringing together data drift detection, concept drift detection, xApp conflict identification, evidence aggregation, cascading-effect analysis, and decision support, the framework aims to move beyond simple anomaly detection toward more interpretable and actionable diagnosis.

The central objective is to determine **what is degrading, why it may be degrading, how the causes may be related, and which corrective action should be evaluated next.**
