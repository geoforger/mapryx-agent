type BillingConfig = {
  workersUsdPer1000Neurons: number;
  planPriceUsd: number;
  includedCredits: number;
  targetGrossMargin: number;
};

type ModelTokenPricing = {
  inputUsdPerMillionTokens: number;
  outputUsdPerMillionTokens: number;
};

const DEFAULT_BILLING_CONFIG: BillingConfig = {
  workersUsdPer1000Neurons: 0.011,
  planPriceUsd: 10,
  includedCredits: 1000,
  targetGrossMargin: 0.7
};

const MODEL_TOKEN_PRICING: Record<string, ModelTokenPricing> = {
  "@cf/zai-org/glm-4.7-flash": {
    inputUsdPerMillionTokens: 0.06,
    outputUsdPerMillionTokens: 0.4
  }
};

const DEFAULT_MODEL_TOKEN_PRICING: ModelTokenPricing = {
  inputUsdPerMillionTokens: 0.06,
  outputUsdPerMillionTokens: 0.4
};

const DEFAULT_RESERVATION_CREDITS = 1;

function getBillingParameters(config: BillingConfig) {
  const allowedCostRatio = 1 - config.targetGrossMargin;
  const usdPerNeuron = config.workersUsdPer1000Neurons / 1000;
  const usdPerCredit =
    (config.planPriceUsd * allowedCostRatio) / config.includedCredits;
  const neuronsPerCredit = usdPerCredit / usdPerNeuron;
  const inputTokensPerCredit =
    usdPerCredit /
    (DEFAULT_MODEL_TOKEN_PRICING.inputUsdPerMillionTokens / 1_000_000);
  const outputTokensPerCredit =
    usdPerCredit /
    (DEFAULT_MODEL_TOKEN_PRICING.outputUsdPerMillionTokens / 1_000_000);

  return {
    usdPerNeuron,
    usdPerCredit,
    neuronsPerCredit,
    inputTokensPerCredit,
    outputTokensPerCredit
  };
}

function getModelTokenPricing(model: string): ModelTokenPricing {
  return MODEL_TOKEN_PRICING[model] ?? DEFAULT_MODEL_TOKEN_PRICING;
}

function calculateTokenCostUsd(
  inputTokens: number,
  outputTokens: number,
  model: string
): number {
  const pricing = getModelTokenPricing(model);

  return (
    inputTokens * (pricing.inputUsdPerMillionTokens / 1_000_000) +
    outputTokens * (pricing.outputUsdPerMillionTokens / 1_000_000)
  );
}

function estimateNeuronsFromTokenCost(
  tokenCostUsd: number,
  config: BillingConfig = DEFAULT_BILLING_CONFIG
): number {
  const { usdPerNeuron } = getBillingParameters(config);
  return Math.ceil(tokenCostUsd / usdPerNeuron);
}

function calculateCreditsFromTokenCost(
  tokenCostUsd: number,
  config: BillingConfig = DEFAULT_BILLING_CONFIG
): number {
  const { usdPerCredit } = getBillingParameters(config);
  return Math.max(1, Math.ceil(tokenCostUsd / usdPerCredit));
}

export class CreditService {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(supabaseUrl: string, serviceRoleKey: string) {
    this.baseUrl = `${supabaseUrl}/rest/v1/rpc`;
    this.headers = {
      "Content-Type": "application/json",
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`
    };
  }

  async reserveCredits(
    userId: string,
    agentRunId: string,
    credits: number = DEFAULT_RESERVATION_CREDITS
  ): Promise<{ success: boolean; reservationId: string | null }> {
    const res = await fetch(`${this.baseUrl}/reserve_credits`, {
      method: "POST",
      headers: { ...this.headers, Prefer: "return=representation" },
      body: JSON.stringify({
        p_user_id: userId,
        p_agent_run_id: agentRunId,
        p_credits: credits
      })
    });

    if (!res.ok) {
      console.error("[credits] reserve failed:", res.status, await res.text());
      return { success: false, reservationId: null };
    }

    const rows = (await res.json()) as Array<{
      success: boolean;
      reservation_id: string | null;
    }>;
    const row = rows[0];
    if (!row || !row.success) {
      return { success: false, reservationId: null };
    }
    return { success: true, reservationId: row.reservation_id };
  }

  async settleUsage(
    userId: string,
    agentRunId: string,
    inputTokens: number,
    outputTokens: number,
    model: string
  ): Promise<void> {
    const { neurons, credits } = this.calculateUsage(
      inputTokens,
      outputTokens,
      model
    );

    const res = await fetch(`${this.baseUrl}/settle_credit_usage`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        p_user_id: userId,
        p_agent_run_id: agentRunId,
        p_actual_credits: credits,
        p_neurons: neurons,
        p_model: model,
        p_input_tokens: inputTokens,
        p_output_tokens: outputTokens
      })
    });

    if (!res.ok) {
      console.error("[credits] settle failed:", res.status, await res.text());
    }
  }

  async releaseReservation(userId: string, agentRunId: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/release_credit_reservation`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        p_user_id: userId,
        p_agent_run_id: agentRunId
      })
    });

    if (!res.ok) {
      console.error("[credits] release failed:", res.status, await res.text());
    }
  }

  calculateUsage(
    inputTokens: number,
    outputTokens: number,
    model: string
  ): { neurons: number; credits: number } {
    // Credits are billed from known token pricing so plan economics are exact
    // for the current model. We still estimate neurons from that same cost for
    // storage/debugging because the SDK path here does not expose actual
    // Workers neuron usage.
    const tokenCostUsd = calculateTokenCostUsd(
      inputTokens,
      outputTokens,
      model
    );
    const neurons = estimateNeuronsFromTokenCost(tokenCostUsd);
    const credits = calculateCreditsFromTokenCost(tokenCostUsd);
    return { neurons, credits };
  }
}

export const DEFAULT_RESERVE_CREDITS = DEFAULT_RESERVATION_CREDITS;
