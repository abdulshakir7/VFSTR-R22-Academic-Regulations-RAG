# 🎓 VFSTR R22 Academic Regulations RAG Assistant

A Retrieval-Augmented Generation (RAG) based academic assistant that answers student queries using the **VFSTR B.Tech R22 Academic Regulations** document.

---

## 👨‍🎓 Student Details

- **Name:** K. Abdul Shukkur
- **Registration No.:** 231FA18067
- **Program:** B.Tech Artificial Intelligence & Machine Learning
- **Section:** C
- **Year:** Final Year — 4-1
- **Institution:** VFSTR

---

## 📌 Project Overview

The VFSTR R22 Academic Regulations RAG Assistant is a document-based question-answering system designed to help students quickly find information from the official VFSTR R22 academic regulations.

Instead of manually searching through a lengthy PDF, users can ask questions in natural language. The system retrieves relevant sections from the regulations and uses Google Gemini to generate an answer based only on the retrieved content.

---

## 🎯 Objectives

- Provide quick access to academic regulations.
- Allow students to ask questions using natural language.
- Retrieve relevant information from the VFSTR R22 regulations.
- Generate concise answers using Google Gemini.
- Display the source pages used to generate the answer.
- Reduce the need for manual PDF searching.

---

## 🧠 System Architecture

```text
VFSTR R22 Regulations PDF
          ↓
     Text Extraction
        (PyPDF)
          ↓
      Text Chunking
          ↓
 Sentence Transformer
    Embeddings
          ↓
      FAISS Index
          ↓
   Relevant Chunks
      Retrieved
          ↓
     Google Gemini
          ↓
    Generated Answer
          ↓
   Source Page Numbers