## 📖 Overview

*Flame in Freefall* is an Artificial Intelligence system designed to ingest, classify, and present NASA’s extensive historical and experimental datasets on *microgravity combustion and spacecraft fire safety* in a clear, structured, and accessible format.

In microgravity environments (such as the International Space Station), fire behaves radically differently than on Earth—forming spherical flames, propagating under unique atmospheric conditions, and posing critical risks to crewed missions. This project leverages trained machine learning models to transform complex telemetry, research papers, and experimental logs into actionable, easy-to-navigate insights.

---

## 🎯 Key Objectives

* *Automated Data Classification:* Categorize decades of unstructured and structured NASA combustion data (e.g., FLEX, Saffire, and Burning and Suppression of Solids experiments).
* *Simplified Knowledge Retrieval:* Translate dense aerospace telemetry and scientific documentation into clean, human-readable summaries and structured taxonomies.
* *Fire Safety Intelligence:* Identify flammability thresholds, material ignition risks, and suppression dynamics to support safer spacecraft design and mission protocols.

---

## 🧠 System Architecture & Workflow

The AI pipeline was designed and trained through the following stages:

1. *Data Ingestion:* Collection of open-source datasets from the NASA Physical Sciences Informatics (PSI) database and technical reports.
2. *Preprocessing & Feature Extraction:* Cleaning telemetry logs (temperature, pressure, $O_2$/$CO_2$ concentrations) and parsing scientific text using Natural Language Processing (NLP).
3. *Model Training & Classification:* Supervised and unsupervised learning algorithms trained to tag experiments by material type, flame geometry, extinction behavior, and risk level.
4. *Structured Presentation Layer:* Outputting organized datasets, visual dashboards, and searchable records for researchers, engineers, and students.

---

## 📂 Repository Structure

```text
flame-in-freefall/
├── data/                   # Raw and processed NASA microgravity datasets
├── models/                 # Trained AI classification weights and architectures
├── notebooks/              # Exploratory Data Analysis (EDA) and training pipelines
├── src/
│   ├── ingestion/          # Scripts for parsing NASA data logs and PDFs
│   ├── classifier/         # Core AI classification and NLP tagging logic
│   └── visualization/      # Tools for clean data presentation and reporting
├── docs/                   # Technical documentation and taxonomy guides
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
