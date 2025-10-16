<template>
  <div class="container">
    <h1>RAG PDF QA System</h1>

    <section>
      <h2>Upload PDFs</h2>
      <input type="file" multiple @change="handleFiles" />
      <button @click="uploadDocuments">Upload</button>
      <p v-if="uploadResult">{{ uploadResult }}</p>
    </section>

    <section>
      <h2>Ask a Question</h2>
      <input type="text" v-model="question" placeholder="Type your question" />
      <button @click="askQuestion">Ask</button>
      <div v-if="answer">
        <h3>Answer:</h3>
        <p>{{ answer }}</p>
        <h4>References:</h4>
        <ul>
          <li v-for="ref in references" :key="ref">{{ ref }}</li>
        </ul>
      </div>
    </section>
  </div>
</template>

<script>
import api from './axios'

export default {
  data() {
    return {
      files: [],
      question: '',
      answer: '',
      references: [],
      uploadResult: ''
    }
  },
  methods: {
    handleFiles(event) {
      this.files = event.target.files
    },
    async uploadDocuments() {
      const formData = new FormData()
      for (let i = 0; i < this.files.length; i++) {
        formData.append('files', this.files[i])
      }
      try {
        const res = await api.post('/api/documents', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        this.uploadResult = `Uploaded: ${res.data.total_chunks} chunks from ${res.data.documents_indexed} documents.`
      } catch (err) {
        console.error(err)
        this.uploadResult = 'Upload failed.'
      }
    },
    async askQuestion() {
      try {
        const res = await api.post('/api/question', { question: this.question })
        this.answer = res.data.answer
        this.references = res.data.references
      } catch (err) {
        console.error(err)
        this.answer = 'Failed to get answer.'
        this.references = []
      }
    }
  }
}
</script>

<style>
.container { max-width: 600px; margin: auto; padding: 2rem; font-family: sans-serif; }
section { margin-bottom: 2rem; }
input[type="text"] { width: 100%; padding: 0.5rem; margin-bottom: 0.5rem; }
button { padding: 0.5rem 1rem; }
</style>
