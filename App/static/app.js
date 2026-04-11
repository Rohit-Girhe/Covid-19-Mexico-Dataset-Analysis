const form = document.getElementById("prediction-form");
const fillSampleButton = document.getElementById("fill-sample");
const submitButton = document.getElementById("submit-button");
const statusBadge = document.getElementById("status-badge");
const probabilityValue = document.getElementById("probability-value");
const probabilityRing = document.getElementById("probability-ring");
const riskLabel = document.getElementById("risk-label");
const riskDescription = document.getElementById("risk-description");
const meterFill = document.getElementById("meter-fill");
const predictedClass = document.getElementById("predicted-class");
const modelVersion = document.getElementById("model-version");
const errorMessage = document.getElementById("error-message");

const threshold = window.APP_CONFIG?.threshold ?? 0.5;

const samplePayload = {
    AGE: 65,
    MEDICAL_UNIT: 1,
    USMER: 0,
    SEX: 1,
    PATIENT_TYPE: 1,
    PNEUMONIA: 1,
    DIABETES: 0,
    COPD: 0,
    ASTHMA: 0,
    INMSUPR: 0,
    HIPERTENSION: 1,
    OTHER_DISEASE: 0,
    CARDIOVASCULAR: 0,
    OBESITY: 0,
    RENAL_CHRONIC: 0,
    TOBACCO: 0
};

function setStatus(mode, text) {
    statusBadge.className = `status-badge ${mode}`;
    statusBadge.textContent = text;
}

function setError(message) {
    errorMessage.hidden = !message;
    errorMessage.textContent = message ?? "";
}

function updateVisuals(probability, label, prediction, version) {
    const percent = Math.max(0, Math.min(100, probability * 100));
    probabilityRing.style.setProperty("--progress", `${percent}%`);
    probabilityValue.textContent = `${percent.toFixed(1)}%`;
    riskLabel.textContent = label;
    predictedClass.textContent = prediction === 1 ? "1 (High Risk)" : "0 (Lower Risk)";
    modelVersion.textContent = version;
    meterFill.style.width = `${percent}%`;

    let tone = "This case is currently below the trained decision threshold. Continue to interpret the score alongside the full clinical picture.";
    if (probability >= threshold) {
        tone = "This case crosses the trained decision threshold and should be treated as elevated mortality risk by the model.";
    }
    if (probability >= Math.max(threshold + 0.2, 0.75)) {
        tone = "The score is deep in the high-risk region. This is the kind of case the model considers strongly concerning.";
    }
    riskDescription.textContent = tone;
}

function collectPayload() {
    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        payload[key] = key === "AGE" ? Number(value) : parseInt(value, 10);
    }

    return payload;
}

function fillSampleValues() {
    Object.entries(samplePayload).forEach(([key, value]) => {
        const field = document.getElementById(key);
        if (field) {
            field.value = String(value);
        }
    });
}

fillSampleButton?.addEventListener("click", fillSampleValues);

form?.addEventListener("reset", () => {
    setStatus("idle", "Waiting for input");
    setError("");
    probabilityValue.textContent = "--";
    probabilityRing.style.setProperty("--progress", "0%");
    riskLabel.textContent = "No prediction yet";
    riskDescription.textContent = "Submit patient details to generate a mortality-risk estimate from the trained ANN model.";
    meterFill.style.width = "0%";
    predictedClass.textContent = "--";
    modelVersion.textContent = "1.0.0";
});

form?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = collectPayload();
    setError("");
    setStatus("loading", "Running prediction");
    submitButton.disabled = true;
    submitButton.textContent = "Predicting...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || "Prediction request failed.");
        }

        updateVisuals(result.probability, result.label, result.prediction, result.model_version);
        setStatus("done", "Prediction complete");
    } catch (error) {
        setStatus("error", "Prediction failed");
        setError(error.message || "Something went wrong while scoring this case.");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Predict Risk";
    }
});
