import {doPost, encodeWAV} from "./util.js";

import axios from "axios";
import "bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import {Modal} from "bootstrap";
import Oruga from "@oruga-ui/oruga-next";
import "@oruga-ui/oruga-next/dist/oruga-full.css";
import "@oruga-ui/oruga-next/dist/oruga-full-vars.css";
import {computed, createApp, nextTick, onMounted, ref} from "vue";
import Slider from "./src/components/Slider.vue";

import AudioMotionAnalyzer from "audiomotion-analyzer";

const app = createApp({
    name: "ChatBot",
    delimiters: ["[[", "]]"],
    components: {
        Slider,
    },
    setup() {
        const session = JSON.parse(document.getElementById("session").textContent);

        const chatHistory = ref(
            [
                {
                    id: 1,
                    content: "You are a helpful assistant.",
                    role: "system",
                },
            ],
        );
        let id = 1;
        const lengthScale = ref(session.length_scale || 5);
        const model = ref({});
        const modelList = ref([]);
        const notice = ref("");
        const prompt = ref("");
        const showMenu = ref(false);
        const speak = ref(session.speak !== undefined ? session.speak : true);
        const temperature = ref(0.7);

        let audioMotion = null;
        const microPhoneOn = ref(false);
        const microPhoneVADOn = ref(false);
        let micStream = null;
        let mediaRecorder;
        let audioChunks = [];
        let myvad = null;

        const sliderSpeed = ref(null);
        const sliderTemperature = ref(null);

        const filteredChatHistory = computed(() => {
            return chatHistory.value.filter((x) => x.role !== "system");
        });

        function addMessage(role, message) {
            id++;
            chatHistory.value.push(
                {
                    id: id,
                    content: message,
                    role: role,
                },
            );
            nextTick(() => {
                const scrollableDiv = document.getElementsByClassName("message-container")[0];
                scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
            });
        };

        // Map values used by the UI, range 1 (slow) to 10 (fast)
        // to values used by Piper TTS, range 0.5 (fast) to 1.5 (slow)
        function mapSpeechRateValue(x) {
            // Source range
            const a = 1;
            const b = 10;

            // Target range
            const c = 1.5;
            const d = 0.5;

            return c + (d - c) * (x - a) / (b - a);
        }

        function getListenButtonValue() {
            return microPhoneOn.value ? "Mic Off" : "Mic On";
        };

        function getMarkdown(content) {
            return markdown.render(content);
        };

        function handleChangeModel(event) {
            const modal = new Modal("#modalProcessing");
            modal.show();
            doPost(
                "/load",
                {
                    "model": event.srcElement.value,
                },
                (response) => {
                    modal.hide();
                },
                "Error loading model",
            );
        };

        function handleSendMessage(event) {
            sendMessageToChatbot(prompt.value);
        };

        function getModelInfo() {
            doPost(
                "/info",
                {
                },
                (response) => {
                    model.value = response.data.response;
                },
                "Error getting model info",
            );
        };

        function getModelList() {
            doPost(
                "/list",
                {
                    "action": "list",
                },
                (response) => {
                    modelList.value = response.data.response;
                },
                "Error getting model info",
            );
        };

        function getOrCreateAudioMotionAnalzyer(gradient) {
            if (!audioMotion) {
                audioMotion = new AudioMotionAnalyzer(
                    document.getElementById("canvas-container"),
                    {
                        bgAlpha: 0,
                        overlay: true,
                        showScaleX: false,
                    },
                );
            }
            audioMotion.gradient = gradient;
            return audioMotion;
        };

        function connectStream(stream) {
            // create stream using audioMotion audio context
            micStream = audioMotion.audioCtx.createMediaStreamSource(stream);
            // connect microphone stream to analyzer
            audioMotion.connectInput(micStream);
            // mute output to prevent feedback loops from the speakers
            audioMotion.volume = 0;
        };

        async function playWav(base64String) {
            // Convert Base64 string to ArrayBuffer
            const byteString = atob(base64String);
            const arrayBuffer = new ArrayBuffer(byteString.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }

            audioMotion = getOrCreateAudioMotionAnalzyer("steelblue");

            // Decode and play the audio, with visualization
            audioMotion.audioCtx.decodeAudioData(arrayBuffer, (audioBuffer) => {
                const source = audioMotion.audioCtx.createBufferSource();
                audioMotion.connectInput(source);
                source.buffer = audioBuffer;
                source.connect(audioMotion.audioCtx.destination);
                source.start();
            });
        }

        async function sendMessageToChatbot(message) {
            addMessage("user", message);
            prompt.value = "";
            notice.value = "Waiting for the AI";
            doPost(
                "/chat",
                {
                    "message": message,
                    "length_scale": mapSpeechRateValue(lengthScale.value),
                    "speak": speak.value,
                    "temperature": temperature.value,
                },
                (response) => {
                    addMessage("assistant", response.data.response);
                    notice.value = "";
                    if (speak.value && response.data.audio) {
                        playWav(response.data.audio);
                    }
                },
                "Error",
            );
        };

        async function handleListen() {
            if (microPhoneOn.value) {
                // disconnect and release microphone stream
                audioMotion.disconnectInput( micStream, true );
                return;
            }

            notice.value = "Listening...";

            audioMotion = getOrCreateAudioMotionAnalzyer("rainbow");

            navigator.mediaDevices.getUserMedia( {audio: true, video: false} )
                .then( (stream) => {
                    connectStream(stream);

                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        notice.value = "";
                        const blob = new Blob(audioChunks, {type: "audio/wav"});
                        const formData = new FormData();
                        formData.append("audio", blob);
                        notice.value = "Waiting for speech to text";
                        axios.post("/speech2text",
                                   formData,
                                   {
                                       headers: {
                                           "Content-Type": "multipart/form-data",
                                       },
                                   }).then((response) => {
                                       sendMessageToChatbot(response.data.input);

                                       // Delete the current audio in case we want to start a new recording later
                                       audioChunks = [];
                                   });
                    };
                })
                .catch( (err) => {
                    alert("Microphone access denied by user: " + err);
                });
        };

        async function handleListenVAD() {
            if (microPhoneVADOn.value) {
                // disconnect and release microphone stream
                audioMotion.disconnectInput( micStream, true );
                myvad.pause();
                return;
            }

            myvad = await vad.MicVAD.new({
                onSpeechStart: () => {
                    notice.value = "Listening...";
                    audioMotion.gradient = "rainbow";
                },
                onSpeechEnd: (audio) => {
                    notice.value = "";
                    audioMotion.gradient = "steelblue";
                    const wavBuffer = encodeWAV(audio);
                    const blob = new Blob([wavBuffer], {type: "audio/wav"});
                    const formData = new FormData();
                    formData.append("audio", blob);
                    notice.value = "Waiting for speech to text";
                    axios.post("/speech2text",
                               formData,
                               {
                                   headers: {
                                       "Content-Type": "multipart/form-data",
                                   },
                               }).then((response) => {
                                   sendMessageToChatbot(response.data.input);
                                   // Delete the current audio in case we want to start a new recording later
                                   audioChunks = [];
                               });
                },
            });
            audioMotion = getOrCreateAudioMotionAnalzyer("rainbow");
            connectStream(myvad.stream);
            myvad.start();
        };

       onMounted(() => {
            const menuDiv = document.getElementById("menu");
            const prefsDiv = document.getElementsByClassName("hamburger")[0];

            document.addEventListener("click", function(event) {
                // Check if the clicked area is menuDiv or a descendant of menuDiv
                const isClickInside = menuDiv.contains(event.target);

                // If the clicked area is outside menuDiv and we're not clicking the
                //  hamburger menu icon, hide it!
                if (!isClickInside && !prefsDiv.contains(event.target)) {
                    showMenu.value = false;
                }
            });

            getModelInfo();
            getModelList();

            document.getElementById("prompt").focus();
        });

        return {
            chatHistory,
            filteredChatHistory,
            handleChangeModel,
            handleListen,
            handleListenVAD,
            handleSendMessage,
            getListenButtonValue,
            getMarkdown,
            lengthScale,
            model,
            modelList,
            microPhoneOn,
            microPhoneVADOn,
            notice,
            prompt,
            showMenu,
            sliderSpeed,
            sliderTemperature,
            speak,
            temperature,
        };
    },
});
app.use(Oruga, {});
app.mount("#vue-app");

import "animate.css";

// Use the tiny-emitter package as an event bus
import emitter from "tiny-emitter/instance";

const EventBus = {
  $on: (...args) => emitter.on(...args),
  $once: (...args) => emitter.once(...args),
  $off: (...args) => emitter.off(...args),
    $emit: (...args) => emitter.emit(...args),
};
window.EventBus = EventBus;

import hljs from "highlight.js";
const markdown = require("markdown-it")({
    highlight: function(str) {
      try {
          return hljs.highlightAuto(str).value;
      } catch (__) {}

    return "";
    },
});
window.markdown = markdown;
import "highlight.js/styles/dracula.css";
