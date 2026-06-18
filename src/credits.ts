const MODEL_NEURON_RATES: Record<string, { input: number; output: number }> = {
  "@cf/zai-org/glm-4.7-flash": { input: 0.01, output: 0.04 },
};

const NEURONS_PER_CREDIT = 1000;
const DEFAULT_RESERVATION_CREDITS = 1;
const DEFAULT_NEURON_RATE = { input: 0.02, output: 0.06 };

export class CreditService {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(supabaseUrl: string, serviceRoleKey: string) {
    this.baseUrl = `${supabaseUrl}/rest/v1/rpc`;
    this.headers = {
      "Content-Type": "application/json",
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
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
        p_credits: credits,
      }),
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
        p_output_tokens: outputTokens,
      }),
    });

    if (!res.ok) {
      console.error("[credits] settle failed:", res.status, await res.text());
    }
  }

  async releaseReservation(
    userId: string,
    agentRunId: string
  ): Promise<void> {
    const res = await fetch(`${this.baseUrl}/release_credit_reservation`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        p_user_id: userId,
        p_agent_run_id: agentRunId,
      }),
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
    const rates = MODEL_NEURON_RATES[model] ?? DEFAULT_NEURON_RATE;
    const neurons = Math.ceil(
      inputTokens * rates.input + outputTokens * rates.output
    );
    const credits = Math.max(1, Math.ceil(neurons / NEURONS_PER_CREDIT));
    return { neurons, credits };
  }
}

export const DEFAULT_RESERVE_CREDITS = DEFAULT_RESERVATION_CREDITS;
