# Mini Agent Swarm: destilando um orquestrador multi-agente para uma SLM

Projeto prático para o grupo de estudos de NLP avançado (tema: multi-agentes).
**Apresentação:** terça-feira, 17/11/2026.

## Ideia

Reproduzir, em escala pequena, a ideia do *Agent Swarm* do Kimi K2.5: um modelo que
atua como **orquestrador**, divide a tarefa e cria **sub-agentes** (cópias de si mesmo)
que trabalham em paralelo, cada um com contexto próprio. Um modelo professor grande
gera trajetórias nesse formato e um modelo aluno pequeno (≤4B) é treinado nelas (destilação).

## Decisões tomadas

| Item | Decisão |
|---|---|
| Abordagem | **A — destilar o orquestrador** |
| Família de modelos | **Qwen** (professor e aluno da mesma família) |
| Geração | Definida pelo **teste de sanidade** (ver abaixo) |
| Candidato 1 (geração nova) | Professor **Qwen3.6-35B-A3B** (Q4) → aluno **Qwen3.5-2B** (upgrade: Qwen3.5-4B) |
| Candidato 2 (fallback) | Professor **Qwen3-30B-A3B** (Q4) → aluno **Qwen3-1.7B** |
| Hardware | 1× RTX 4090 (24 GB). Professor e treino rodam em fases separadas |
| Orçamento | **R$ 0**: tudo local, sem APIs pagas |
| Treino | SFT com LoRA (Unsloth/TRL), com loss mask nos retornos de tools. RL (GRPO) fica fora do escopo |

## Teste de sanidade (decide a geração)

Com cerca de 30 exemplos no harness, medir para cada candidato:
- o professor cabe na VRAM, e com qual contexto máximo;
- tokens/s, em requisição única e em lote;
- % de tool calls com formato válido;
- % de respostas corretas.

Critério: usar a geração nova, a menos que ela quebre (suporte de bibliotecas,
memória) ou fique claramente pior nas métricas acima.

## Objetivo / tarefa

MedCalc-Bench Verified, calculadoras baseadas em regras. O orquestrador divide as variáveis em grupos,
cria sub-agentes extratores em paralelo e chama `calculate()` (implementação oficial).
Controle principal: o mesmo modelo como agente único. Evidências: `reports/Plano orquestrador MedCalc destilado.md`.

## Status (15/11)

- Feito sem GPU: código (`src/medswarm`), oráculo (96,4% treino / 93,7% teste consistentes), spec, harness,
  filtro de SFT, treino, avaliação, demo, scripts do servidor. Testes com LLM falso passando.
- Próximo passo: configurar a chave SSH e seguir o roteiro de GPU do `README.md`.
- Decisão: apresentar na terça o que estiver pronto.
