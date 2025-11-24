/*
  Silero VAD AudioWorklet Processor
  用于处理音频流：降采样至 16000Hz 并分帧发送给主线程
*/

class Resampler {
  constructor(fromSampleRate, toSampleRate, channels) {
    this.fromSampleRate = fromSampleRate;
    this.toSampleRate = toSampleRate;
    this.channels = channels || 0;
    this.initialize();
  }

  initialize() {
    if (this.fromSampleRate > 0 && this.toSampleRate > 0 && this.channels > 0) {
      if (this.fromSampleRate == this.toSampleRate) {
        this.resampler = (buffer) => {
          return buffer;
        };
        this.ratio = 1;
      } else {
        this.ratio = this.fromSampleRate / this.toSampleRate;
        this.resampler = (buffer) => {
          const bufferLength = buffer.length;
          const outputLength = Math.round(bufferLength / this.ratio);
          const result = new Float32Array(outputLength);
          let offsetResult = 0;
          let offsetBuffer = 0;
          while (offsetResult < outputLength) {
            let nextOffsetBuffer = Math.round((offsetResult + 1) * this.ratio);
            let accum = 0,
              count = 0;
            for (
              let i = offsetBuffer;
              i < nextOffsetBuffer && i < bufferLength;
              i++
            ) {
              accum += buffer[i];
              count++;
            }
            result[offsetResult] = accum / count;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
          }
          return result;
        };
      }
    }
  }

  process(buffer) {
    return this.resampler(buffer);
  }
}

class SileroVadProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this._frameSize = options.processorOptions.frameSize || 1536; // 96ms @ 16kHz
    this._buffer = [];
    this._resampler = null;
    this._sourceSampleRate = options.processorOptions.sourceSampleRate || 44100;
    
    // Silero VAD 需要 16000Hz 采样率
    if (this._sourceSampleRate !== 16000) {
        this._resampler = new Resampler(this._sourceSampleRate, 16000, 1);
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channel0 = input[0]; // 仅处理单声道
      
      // 1. 降采样 (如果需要)
      let processedData = channel0;
      if (this._resampler) {
          processedData = this._resampler.process(channel0);
      }

      // 2. 缓冲数据
      for (let i = 0; i < processedData.length; i++) {
        this._buffer.push(processedData[i]);
      }

      // 3. 分帧发送 (当缓冲区满 frameSize 时)
      while (this._buffer.length >= this._frameSize) {
        const frame = this._buffer.slice(0, this._frameSize);
        this._buffer = this._buffer.slice(this._frameSize);
        
        // 发送消息给主线程 (vad-web 监听 "message" 事件)
        this.port.postMessage({ 
            message: 'PROCESS', 
            inputFrame: Float32Array.from(frame) 
        });
      }
    }
    return true;
  }
}

// 注册 Processor，名称需与主线程调用时一致
registerProcessor('vad-helper-worklet', SileroVadProcessor);