use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;

/// Request body for the RunPod vLLM endpoint
#[derive(Debug, Serialize)]
pub struct VllmRequest {
    pub input: VllmInput,
}

#[derive(Debug, Serialize)]
pub struct VllmInput {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,
}

/// Sampling parameters for vLLM generation
#[derive(Debug, Serialize)]
pub struct SamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_json: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Response from the RunPod vLLM endpoint
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VllmResponse {
    pub delay_time: u64,
    pub execution_time: u64,
    pub id: String,
    pub output: Vec<VllmOutput>,
    pub status: String,
    pub worker_id: String,
}

#[derive(Debug, Deserialize)]
pub struct VllmOutput {
    pub choices: Vec<VllmChoice>,
    pub usage: VllmUsage,
}

#[derive(Debug, Deserialize)]
pub struct VllmChoice {
    pub tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct VllmUsage {
    pub input: u32,
    pub output: u32,
}

/// Client for interacting with RunPod vLLM endpoint
pub struct VllmClient {
    api_key: String,
    endpoint_id: String,
    client: reqwest::Client,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

impl VllmClient {
    /// Create a new VllmClient from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key =
            env::var("RUNPOD_API_KEY").context("RUNPOD_API_KEY not found in environment")?;
        let endpoint_id = env::var("RUNPOD_VLLM_ENDPOINT_ID")
            .context("RUNPOD_VLLM_ENDPOINT_ID not found in environment")?;

        Ok(Self {
            api_key,
            endpoint_id,
            client: reqwest::Client::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
        })
    }

    /// Get the API endpoint URL
    fn endpoint_url(&self) -> String {
        format!("https://api.runpod.ai/v2/{}/runsync", self.endpoint_id)
    }

    /// Send a prompt to the vLLM endpoint and get the response
    pub async fn generate(&mut self, prompt: String) -> Result<VllmResponse> {
        self.generate_with_params(prompt, None, None, None, None)
            .await
    }

    /// Send a prompt with custom generation parameters
    pub async fn generate_with_params(
        &mut self,
        prompt: String,
        guided_json: Option<serde_json::Value>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        top_p: Option<f32>,
    ) -> Result<VllmResponse> {
        let sampling_params = if guided_json.is_some() || temperature.is_some() || max_tokens.is_some() || top_p.is_some() {
            Some(SamplingParams {
                guided_json,
                temperature,
                max_tokens,
                top_p,
            })
        } else {
            None
        };

        let request = VllmRequest {
            input: VllmInput {
                prompt,
                sampling_params,
            },
        };

        let response = self
            .client
            .post(&self.endpoint_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to RunPod API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API request failed with status {}: {}", status, error_text);
        }

        // Get the response text first for debugging
        let response_text = response.text().await
            .context("Failed to read response text from RunPod API")?;

        // Log the raw response for debugging
        eprintln!("[DEBUG] Raw API response: {}", response_text);

        // Try to parse the response
        let vllm_response = serde_json::from_str::<VllmResponse>(&response_text)
            .context(format!("Failed to parse response from RunPod API. Response was: {}", response_text))?;

        // Track token usage
        for output in &vllm_response.output {
            self.total_input_tokens += output.usage.input as u64;
            self.total_output_tokens += output.usage.output as u64;
        }

        Ok(vllm_response)
    }

    /// Extract the generated text from the response
    pub fn extract_text(response: &VllmResponse) -> String {
        response
            .output
            .iter()
            .flat_map(|output| &output.choices)
            .flat_map(|choice| &choice.tokens)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get total tokens used
    pub fn get_token_stats(&self) -> (u64, u64) {
        (self.total_input_tokens, self.total_output_tokens)
    }
}
