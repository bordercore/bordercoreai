import {doGet, doPost, encodeWAV} from "./util.js";

import axios from "axios";
import "bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import {library} from "@fortawesome/fontawesome-svg-core";
import {FontAwesomeIcon} from "@fortawesome/vue-fontawesome";
import {faPlus, faRotateLeft} from "@fortawesome/free-solid-svg-icons";
library.add(faPlus, faRotateLeft);
import {Modal} from "bootstrap";
import Oruga from "@oruga-ui/oruga-next";
import "@oruga-ui/oruga-next/dist/oruga-full.css";
import "@oruga-ui/oruga-next/dist/oruga-full-vars.css";
import {computed, createApp, nextTick, onMounted, ref} from "vue";
import Slider from "./src/components/Slider.vue";

import AudioMotionAnalyzer from "audiomotion-analyzer";

const EventBus = {
    $on: (...args) => emitter.on(...args),
    $once: (...args) => emitter.once(...args),
    $off: (...args) => emitter.off(...args),
    $emit: (...args) => emitter.emit(...args),
};
window.EventBus = EventBus;

const app = createApp({
    name: "ChatBot",
    delimiters: ["[[", "]]"],
    components: {
        FontAwesomeIcon,
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
        const controlLights = ref(false);
        const error = ref("");
        const audioSpeed = ref(session.audio_speed || 1);
        const model = ref({});
        const modelList = ref([]);
        const notice = ref("");
        const prompt = ref("");
        const showMenu = ref(false);
        const speak = ref(session.speak !== undefined ? session.speak : true);
        const temperature = ref(0.7);

        const tts = "alltalk";
        let tts_host = ref("10.3.2.5:7851")
        const audioElement = new Audio();
        audioElement.crossOrigin = "anonymous";
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
                    setTimeout(function(){
                        modal.hide();
                    }, 500);
                    if (response.status !== "OK") {
                    }
                },
                "",
                () => {
                    setTimeout(function(){
                        modal.hide();
                    }, 500);
                }
            );
        };

        function handleNewChat(event) {
            chatHistory.value.length = 1;
        };

        function handleRegenerate(event) {
            sendMessageToChatbot(prompt.value, true);
        };

        function handleSendMessage(event) {
            sendMessageToChatbot(prompt.value);
        };

        function getModelInfo() {
            doGet(
                "/info",
                (response) => {
                    model.value = response.data;
                },
                "Error getting model info",
            );
        };

        function getModelList() {
            doGet(
                "/list",
                (response) => {
                    modelList.value = response.data;
                },
                "Error getting model list",
            );
        };

        function createAudioMotionAnalyzer(audioElement) {
            audioMotion = new AudioMotionAnalyzer(
                document.getElementById("canvas-container"),
                {
                    bgAlpha: 0,
                    overlay: true,
                    showScaleX: false,
                    source: audioElement,
                },
            );
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

            audioMotion.gradient = "steelblue";

            // Decode and play the audio, with visualization
            audioMotion.audioCtx.decodeAudioData(arrayBuffer, (audioBuffer) => {
                const source = audioMotion.audioCtx.createBufferSource();
                audioMotion.connectInput(source);
                source.buffer = audioBuffer;
                source.connect(audioMotion.audioCtx.destination);
                source.start();
            });
        }

        async function sendMessageToChatbot(message, regenerate=false) {
            // If we're regenerating the response, remove the last response from
            //   chatHistory and resubmit everything else to the AI.
            if (regenerate) {
                chatHistory.value.pop();
            } else {
                addMessage("user", message);
            }
            prompt.value = "";
            notice.value = "Waiting for the AI";
            doPost(
                "/chat",
                {
                    "message": JSON.stringify(chatHistory.value),
                    "audio_speed": audioSpeed.value,
                    "speak": speak.value,
                    "tts": tts,
                    "temperature": temperature.value,
                    "control_lights": controlLights.value,
                },
                (response) => {
                    addMessage("assistant", response.data.response);
                    console.log(`Speed: ${response.data.speed} t/s`);
                    notice.value = "";

                    if (!speak.value) {
                        return;
                    }

                    if (tts === "alltalk") {
                        const voice = "valerie.wav";
                        const outputFile = "stream_output.wav";
                        const streamingUrl = `http://${tts_host.value}/api/tts-generate-streaming?text=${response.data.response}&voice=${voice}&language=en&output_file=${outputFile}`;
                        audioElement.src = streamingUrl;
                        audioMotion.gradient = "steelblue";
                        audioMotion.volume = 1;
                        audioElement.playbackRate = audioSpeed.value;
                        audioElement.play();
                    } else if (response.data.audio) {
                        playWav(response.data.audio);
                    }

                },
                "",
            );
        };

        async function handleListen() {
            if (microPhoneOn.value) {
                // disconnect and release microphone stream
                audioMotion.disconnectInput( micStream, true );
                return;
            }

            notice.value = "Listening...";

            audioMotion.gradient = "rainbow";

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
                    audioMotion.volume = 0;
                },
                onSpeechEnd: (audio) => {
                    notice.value = "";
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
            audioMotion.gradient = "rainbow";
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

           createAudioMotionAnalyzer(audioElement);

           EventBus.$on("toast", (payload) => {
               error.value = payload;
           });

           getModelInfo();
           getModelList();

           document.getElementById("prompt").focus();
        });

        return {
            chatHistory,
            controlLights,
            error,
            filteredChatHistory,
            handleChangeModel,
            handleListen,
            handleListenVAD,
            handleNewChat,
            handleRegenerate,
            handleSendMessage,
            getListenButtonValue,
            getMarkdown,
            audioSpeed,
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
            tts_host
        };
    },
});
app.use(Oruga, {});
app.mount("#vue-app");

import "animate.css";

// Use the tiny-emitter package as an event bus
import emitter from "tiny-emitter/instance";

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
