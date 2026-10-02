---
created: 2026-09-15
updated: 2026-09-20
title: Embeddings y búsqueda semántica
tags:
  - nota
  - inteligencia-artificial
  - embeddings
  - estado/seedling
status: activo
---

# Embeddings y búsqueda semántica

Los embeddings son vectores numéricos que representan el significado
de un texto. Ver [[RAG — Retrieval-Augmented Generation]] para más info.

## Modelos recomendados

| Modelo | Tamaño | Calidad |
|--------|--------|---------|
| nomic-embed-text | 274MB | Muy buena |
| all-minilm | 45MB | Buena |

El modelo [[nomic-embed-text|nomic]] es el más equilibrado para CPU.

## Aplicaciones

La búsqueda semántica permite encontrar documentos por significado,
no por palabras exactas. Si buscas "vacaciones", puedes encontrar
un texto sobre "días de descanso laboral".
