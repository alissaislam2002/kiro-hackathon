class Pcm16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputSamples = [];
    this.position = 0;
    this.outputSamples = [];
    this.ratio = sampleRate / 16000;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input?.length) return true;
    for (let index = 0; index < input.length; index += 1) {
      this.inputSamples.push(input[index]);
    }

    while (this.position + this.ratio <= this.inputSamples.length) {
      const start = Math.floor(this.position);
      const end = Math.max(start + 1, Math.floor(this.position + this.ratio));
      let total = 0;
      for (let index = start; index < end; index += 1) total += this.inputSamples[index];
      const sample = Math.max(-1, Math.min(1, total / (end - start)));
      this.outputSamples.push(sample < 0 ? sample * 0x8000 : sample * 0x7fff);
      this.position += this.ratio;
    }

    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.inputSamples.splice(0, consumed);
      this.position -= consumed;
    }

    while (this.outputSamples.length >= 1600) {
      const pcm = new Int16Array(1600);
      for (let index = 0; index < pcm.length; index += 1) pcm[index] = this.outputSamples[index];
      this.outputSamples.splice(0, pcm.length);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-16k-processor", Pcm16kProcessor);
