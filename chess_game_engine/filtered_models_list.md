# Filtered Models List
## Models from Fireworks, DeepSeek-AI, and OpenAI (excluding Groq, Cerebras, Novita)

### DeepSeek-AI Models (via DeepInfra or Fireworks)

1. **deepseek-ai/deepseek-r1-0528-turbo**
   - Providers: deepinfra, deepseek-ai
   - Context: 33K
   - Pricing: $1/M in, $3/M out

2. **deepseek-ai/deepseek-r1-turbo**
   - Providers: deepinfra, deepseek-ai
   - Context: 41K
   - Pricing: $1/M in, $3/M out

3. **deepseek-ai/deepseek-v3**
   - Providers: deepinfra, deepseek-ai
   - Context: 164K
   - Pricing: $0.38/M in, $0.89/M out

4. **deepseek-ai/deepseek**
   - Providers: fireworks
   - Context: 131K
   - Pricing: $0.56/M in, $1.68/M out

5. **deepseek-ai/deepseek-prover**
   - Providers: fireworks
   - Context: 164K
   - Pricing: $0.9/M in, $0.9/M out

6. **deepseek-ai/deepseek-r1**
   - Providers: deepinfra, fireworks
   - Context: 164K
   - Pricing: $1.35/M in, $5.4/M out

7. **deepseek-ai/deepseek-r1-0528**
   - Providers: deepinfra, fireworks
   - Context: 164K
   - Pricing: $1.35/M in, $5.4/M out

8. **deepseek-ai/deepseek-r1-0528-fast**
   - Providers: nebius
   - Context: 33K
   - Pricing: $2/M in, $6/M out
   - Note: Only nebius (not in preferred providers, but listed for completeness)

### OpenAI Models (via DeepInfra or Fireworks)

9. **openai/gpt-oss-120b-turbo**
   - Providers: deepinfra
   - Context: 131K
   - Pricing: $0.15/M in, $0.6/M out

10. **openai/gpt-oss-20b-eagle3**
    - Providers: fireworks
    - Context: 0 ctx
    - Pricing: $0.07/M in, $0.3/M out

11. **openai/gpt-oss-safeguard-20b**
    - Providers: fireworks
    - Context: 131K
    - Pricing: $0.5/M in, $0.5/M out

12. **openai/gpt-oss-120b**
    - Providers: cerebras, deepinfra, fireworks, groq
    - Context: 131K
    - Pricing: $0.15/M in, $0.6/M out
    - Note: Has cerebras and groq, but also has fireworks and deepinfra

13. **openai/gpt-oss-20b**
    - Providers: deepinfra, fireworks, groq
    - Context: 131K
    - Pricing: $0.07/M in, $0.3/M out
    - Note: Has groq, but also has fireworks and deepinfra

### Fireworks Models

14. **fireworks/devstral-small-2505**
    - Providers: fireworks
    - Context: 131K
    - Pricing: $0.9/M in, $0.9/M out

15. **fireworks/internvl3-78b**
    - Providers: fireworks
    - Context: 16K
    - Pricing: $0.9/M in, $0.9/M out

16. **fireworks/mythomax-l2-13b**
    - Providers: deepinfra, fireworks, novita
    - Context: 4K
    - Pricing: $0.2/M in, $0.2/M out
    - Note: Has novita, but also has fireworks and deepinfra

17. **fireworks/phi-3-mini-128k-instruct**
    - Providers: fireworks
    - Context: 131K
    - Pricing: $0.1/M in, $0.1/M out

---

## Summary

**Pure matches (only preferred providers, no Groq/Cerebras/Novita):**
- deepseek-ai/deepseek-r1-0528-turbo (deepinfra, deepseek-ai)
- deepseek-ai/deepseek-r1-turbo (deepinfra, deepseek-ai)
- deepseek-ai/deepseek-v3 (deepinfra, deepseek-ai)
- deepseek-ai/deepseek (fireworks)
- deepseek-ai/deepseek-prover (fireworks)
- deepseek-ai/deepseek-r1 (deepinfra, fireworks)
- deepseek-ai/deepseek-r1-0528 (deepinfra, fireworks)
- openai/gpt-oss-120b-turbo (deepinfra)
- openai/gpt-oss-20b-eagle3 (fireworks)
- openai/gpt-oss-safeguard-20b (fireworks)
- fireworks/devstral-small-2505 (fireworks)
- fireworks/internvl3-78b (fireworks)
- fireworks/phi-3-mini-128k-instruct (fireworks)

**Partial matches (have preferred providers but also have Groq/Cerebras/Novita):**
- openai/gpt-oss-120b (has cerebras, groq, but also fireworks, deepinfra)
- openai/gpt-oss-20b (has groq, but also fireworks, deepinfra)
- fireworks/mythomax-l2-13b (has novita, but also fireworks, deepinfra)




