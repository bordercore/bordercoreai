import {doGet, doPost} from "./util.js";

import axios from "axios";
import {computed, createApp, h, nextTick, onMounted, onUnmounted, reactive, ref, watch} from "vue";

const vm = createApp({
    name: "ChatBot",
    delimiters: ["[[", "]]"],
    setup() {
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
        const notice = ref("");
        const prompt = ref("");

        let audioContext = null;

        const microPhoneOn = ref(false);
        let mediaRecorder;
        let audioChunks = [];
        let encoder;
        let constraints = { audio: true };

        const filteredChatHistory = computed(() => {
            return chatHistory.value.filter((x) => x.role !== "system");
        });

        function addMessage(role, message) {
            id = id + 1;
            chatHistory.value.push(
                {
                    id: id,
                    content: message,
                    role: role,
                }
            );
            nextTick(() => {
                const scrollableDiv = document.getElementsByClassName("message-container")[0];
                scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
            });
        };

        function getListenButtonValue() {
            return microPhoneOn.value ? "Stop" : "Listen";
        };

        function getMarkdown(content) {
            return markdown.render(content);
        };

        function handleListen() {
            if (microPhoneOn.value) {
                mediaRecorder.stop();
                audioContext.close();
                microPhoneOn.value = false;
                return;
            }
            startRecording();
        };

        function handleSendMessage(event) {
            sendMessageToChatbot(prompt.value);
        };

        async function sendMessageToChatbot(message) {
            addMessage("user", message);
            prompt.value = "";
            notice.value = "Waiting for the AI";
            doGet(
                `/chat?message=${message}`,
                (response) => {
                    addMessage("assistant", response.data.response);
                    notice.value = "";
                },
                "Error!!"
            );
        };

        function startRecording() {
            notice.value = "Listening...";
            microPhoneOn.value = true;
            navigator.mediaDevices.getUserMedia(constraints).then(stream => {
                audioVis(stream);
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    notice.value = "";
                    let blob = new Blob(audioChunks, { type: "audio/wav" });
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

                mediaRecorder.start();
            }).catch(error => {
                console.error("Error accessing the microphone: ", error);
            });
        };

        // Source: https://codepen.io/nfj525/pen/rVBaab
        function audioVis(stream) {
            audioContext = new AudioContext();
            var src = audioContext.createMediaStreamSource(stream);
            var analyser = audioContext.createAnalyser();

            var canvas = document.getElementById("canvas");
            //canvas.width = window.innerWidth;
            //canvas.height = window.innerHeight;
            var ctx = canvas.getContext("2d");

            src.connect(analyser);
            // Commenting on this line mutes the audio
            // analyser.connect(audioContext.destination);

            analyser.fftSize = 256;

            var bufferLength = analyser.frequencyBinCount;

            var dataArray = new Uint8Array(bufferLength);

            var WIDTH = canvas.width;
            var HEIGHT = canvas.height;

            var barWidth = (WIDTH / bufferLength) * 2.5;
            var barHeight;
            var x = 0;

            const body = document.body;
            const bgColor = window.getComputedStyle(body).getPropertyValue("background-color");

            function renderFrame() {
                requestAnimationFrame(renderFrame);

                x = 0;

                analyser.getByteFrequencyData(dataArray);

                ctx.fillStyle = bgColor;
                ctx.fillRect(0, 0, WIDTH, HEIGHT);

                for (let i = 0; i < bufferLength; i++) {
                    barHeight = dataArray[i];

                    let r = barHeight + (25 * (i/bufferLength));
                    let g = 250 * (i/bufferLength);
                    let b = 50;

                    ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
                    ctx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);

                    x += barWidth + 1;
                }
            }

            renderFrame();
        };

        return {
            chatHistory,
            filteredChatHistory,
            handleListen,
            handleSendMessage,
            getListenButtonValue,
            getMarkdown,
            notice,
            prompt,
        };
    }
});
vm.mount("#vue-app");

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
