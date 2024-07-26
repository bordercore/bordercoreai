import {doGet, doPost, encodeWAV} from "./util.js";

import axios from "axios";
import "bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import {library} from "@fortawesome/fontawesome-svg-core";
import {FontAwesomeIcon} from "@fortawesome/vue-fontawesome";
import {faCheck, faCopy, faFileAlt, faPaperclip, faPaste, faPlus, faRotateLeft} from "@fortawesome/free-solid-svg-icons";
library.add(faCheck, faCopy, faFileAlt, faPaperclip, faPaste, faPlus, faRotateLeft);
import "media-chrome";
import {Modal} from "bootstrap";
import Oruga from "@oruga-ui/oruga-next";
import "@oruga-ui/oruga-next/dist/oruga-full.css";
import "@oruga-ui/oruga-next/dist/oruga-full-vars.css";
import {computed, createApp, nextTick, onMounted, ref} from "vue";
import Slider from "./src/components/Slider.vue";

import AudioMotionAnalyzer from "audiomotion-analyzer";

// Vue composables
import useEvent from "./useEvent.js";

const EventBus = {
    $on: (...args) => emitter.on(...args),
    $once: (...args) => emitter.once(...args),
    $off: (...args) => emitter.off(...args),
    $emit: (...args) => emitter.emit(...args),
};
window.EventBus = EventBus;

import Nav from "./vue/Nav.vue";
window.MyNav = Nav;

const app = createApp({
    name: "ChatBot",
    delimiters: ["[[", "]]"],
    components: {
        FontAwesomeIcon,
        MyNav,
        Slider,
    },
    setup() {
        const session = JSON.parse(document.getElementById("session").textContent);
        const settings = JSON.parse(document.getElementById("settings").textContent);

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
        const error = ref("");
        const uploadedFilename = ref(null);
        const audioFileSize = ref(null);
        const audioFileTranscript = ref(null);
        const audioIsPlayingOrPaused = ref(false);
        const audioSpeed = ref(session.audio_speed || 1);
        const clipboard = ref(null);
        const icon = ref("copy");
        const model = ref({});
        const modelList = ref([]);
        const musicInfo = ref(null);
        const currentSong = ref({});
        const notice = ref("");
        const prompt = ref("");
        const ragFileUploaded = ref(false);
        const ragFileSize = ref(null);
        const sha1sum = ref("");
        const showMenu = ref(false);
        const speak = ref(session.speak !== undefined ? session.speak : true);
        const temperature = ref(session.temperature || 0.7);
        const waiting = ref(false);

        const tts = "alltalk";
        const ttsHost = ref(session.tts_host);
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

        useEvent("ended", handleAudioEnded, {id: "player"});
        useEvent("pause", handleAudioPlayerPause, {id: "player"});
        useEvent("play", handleAudioPlayerPlay, {id: "player"});
        useEvent("paste", handlePaste, {});

        const filteredChatHistory = computed(() => {
            return chatHistory.value.filter((x) => x.role !== "system");
        });

        const songIndex = computed(() => {
            return musicInfo.value.findIndex((x) => x === currentSong.value);
        });

        const chatHandlers = {handleSendMessageAudio, sendMessageToChatbotRag, sendMessageToChatbot};

        function addClipboardToMessages() {
            if (!clipboard.value) {
                return chatHistory.value;
            }
            // Deep copy
            const copiedArray = JSON.parse(JSON.stringify(chatHistory.value));

            copiedArray.forEach((element) => {
                if (element.id === clipboard.value.id) {
                    element.content += ": " + clipboard.value.content;
                }
            });
            return copiedArray;
        };

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

        function getModelType(modelName) {
            const result = modelList.value.find((obj) => obj.model === modelName);
            return result ? result.type : "";
        };

        function handleChangeModel(event) {
            const modelType = getModelType(event.srcElement.value);
            let modal = null;
            if (modelType !== "api") {
                modal = new Modal("#modalProcessing");
                modal.show();
            }
            doPost(
                "/load",
                {
                    "model": event.srcElement.value,
                },
                (response) => {
                    if (modelType !== "api") {
                        setTimeout(function() {
                            modal.hide();
                        }, 500);
                    }
                },
                "",
                () => {
                    if (modelType !== "api") {
                        setTimeout(function() {
                            modal.hide();
                        }, 500);
                    }
                },
            );
        };

        function handleClipboardClick(event) {
            const modal = new Modal("#modalClipboard");
            modal.show();
        };

        function handleDeleteClipboard(event) {
            const modal = Modal.getInstance(document.getElementById("modalClipboard"));
            modal.hide();
            setTimeout(function() {
                // Adding a pause avoids a minor layout shift
                clipboard.value = null;
            }, 500);
        };

        function handleFileUpload(event) {
            const modal = new Modal("#modalProcessing");
            modal.show();

            const formData = new FormData();
            const fileData = event.target.files[0];
            if (!fileData) {
                return;
            }
            ragFileSize.value = fileData.size;
            formData.append("file", fileData);
            axios.post(
                "rag/upload",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }).then((response) => {
                    ragFileUploaded.value = true;
                    sha1sum.value = response.data.sha1sum;
                    uploadedFilename.value = event.target.files[0].name;
                    // Wait a tiny amount before closing the modal. If the file upload happens
                    //  really fast, then "hide" could be called before the modal is created
                    window.setTimeout(() => {
                        const modal = Modal.getInstance(document.getElementById("modalProcessing"));
                        modal.hide();
                    }, 500);
                });
        };

        function handleFileUploadAudio(event) {
            const modal = new Modal("#modalProcessing");
            modal.show();

            const formData = new FormData();
            const fileData = event.target.files[0];
            if (!fileData) {
                return;
            }
            formData.append("file", fileData);
            axios.post(
                "audio/upload",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }).then((response) => {
                    audioFileTranscript.value = response.data.text;
                    audioFileSize.value = audioFileTranscript.value.length;
                    uploadedFilename.value = event.target.files[0].name;
                    modal.hide();
                    // Load the audio into the player
                    let el = document.getElementById("audioPlayer");
                    el.classList.replace("d-none", "d-flex");
                    el = document.getElementById("player");
                    const audioURL = URL.createObjectURL(fileData);
                    el.src = audioURL;
                });
        };

        function handleNewChat(event) {
            chatHistory.value.length = 1;
            clipboard.value = null;
        };

        function handleSendMessageRag(event) {
            sendMessageToChatbotRag(prompt.value);
        };

        function sendMessageToChatbotRag(message) {
            setTimeout(function() {
                waiting.value = true;
            }, 500);
            addMessage("user", message);
            prompt.value = "";
            doPost(
                "/rag/chat",
                {
                    "message": message,
                    "sha1sum": sha1sum.value,
                    "audio_speed": audioSpeed.value,
                    "speak": speak.value,
                    "tts": tts,
                },
                (response) => {
                    waiting.value = false;
                    addMessage("assistant", response.data.response);
                    notice.value = "";
                    doTTS(response.data);
                },
                "",
            );
        };

        function handleSendMessageAudio() {
            setTimeout(function() {
                waiting.value = true;
            }, 500);
            const message = prompt.value;
            addMessage("user", message);
            prompt.value = "";
            doPost(
                "/audio/chat",
                {
                    "message": message,
                    "transcript": audioFileTranscript.value,
                    "model": model.value,
                    "sha1sum": sha1sum.value,
                    "audio_speed": audioSpeed.value,
                    "speak": speak.value,
                    "tts": tts,
                    "temperature": temperature.value,
                },
                (response) => {
                    waiting.value = false;
                    addMessage("assistant", response.data.content);
                    console.log(`Speed: ${response.data.speed} t/s`);
                    notice.value = "";
                    doTTS(response.data);
                },
                "",
            );
        };

        function handleCopyText(event) {
            if (navigator.clipboard) {
                navigator.clipboard.writeText(audioFileTranscript.value);

                icon.value = "check";
                setTimeout(() => {
                    icon.value = "copy";
                }, 2000);
            }
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
            setTimeout(function() {
                waiting.value = true;
            }, 500);
            // If we're regenerating the response, remove the last response from
            //   chatHistory and resubmit everything else to the AI.
            if (regenerate) {
                chatHistory.value.pop();
            } else {
                addMessage("user", message);
            }
            const messages = addClipboardToMessages();
            prompt.value = "";
            doPost(
                "/chat",
                {
                    "message": JSON.stringify(messages),
                    "model": model.value,
                    "audio_speed": audioSpeed.value,
                    "speak": speak.value,
                    "tts": tts,
                    "temperature": temperature.value,
                },
                (response) => {
                    waiting.value = false;
                    if (response.data?.music_info?.length > 0) {
                        musicInfo.value = response.data.music_info;
                        playSong(musicInfo.value[0]);
                    }
                    addMessage("assistant", response.data.content);
                    console.log(`Speed: ${response.data.speed} t/s`);
                    doTTS(response.data);
                },
                "",
            );
        };

        async function handleListen(chatHandler) {
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
                                chatHandlers[chatHandler](response.data.input);
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

        function doTTS(response) {
            if (!speak.value) {
                return;
            }

            if (tts === "alltalk") {
                const voice = session.tts_voice;
                const outputFile = "stream_output.wav";
                const streamingUrl = `http://${ttsHost.value}/api/tts-generate-streaming?text=${response.content}&voice=${voice}&language=en&output_file=${outputFile}`;
                audioElement.src = streamingUrl;
                audioMotion.gradient = "steelblue";
                audioMotion.volume = 1;
                audioElement.playbackRate = audioSpeed.value;
                audioElement.play();
            } else if (response.audio) {
                playWav(response.audio);
            }
        }

        function handleAudioPlayerPlay(event) {
            if (audioIsPlayingOrPaused.value) {
                // If this is already set, then we're probably playing
                //  from a paused state, so change the animated icon.
                const src = "/static/img/equaliser-animated-green.gif";
                document.getElementById("isPlaying").src = src;
            } else {
                audioIsPlayingOrPaused.value = true;
            }
        };

        function handleAudioPlayerPause(event) {
            const src = "/static/img/equaliser-animated-green-frozen.gif";
            document.getElementById("isPlaying").src = src;
        };

        function handlePaste(event) {
            event.preventDefault();
            const paste = (event.clipboardData || window.clipboardData).getData("text");

            if (paste.length > 200) {
                clipboard.value = {"content": paste, "id": id + 1};
            } else {
                prompt.value += paste;
            }
        };

        async function loadOptionalModule(uuid) {
            return await import(/* webpackChunkName: "optional" */ "@optional-module");
            module.run(uuid);
        }

        async function playSong(song) {
            let el = document.getElementById("audioPlayer");
            el.classList.replace("d-none", "d-flex");
            el = document.getElementById("player");
            // Play the first song
            currentSong.value = song;
            el.src = settings.music_uri + song.uuid;
            el.play();

            const myModule = await loadOptionalModule(song.uuid);
            myModule.run(song.uuid);
        };

        function handleAudioEnded() {
            if (songIndex.value < musicInfo.value.length) {
                playSong(musicInfo.value[songIndex.value + 1]);
            } else {
                musicInfo.value = null;
            }
        };

        function getAudioFileSize() {
            return formatBytes(audioFileSize.value);
        };

        function getRagFileSize() {
            return formatBytes(ragFileSize.value);
        };

        function formatBytes(bytes, decimals = 2) {
            if (!+bytes) return "0 Bytes";

            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ["b", "k", "M", "G", "T", "P", "E", "Z", "Y"];

            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return `${parseInt((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
        }

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
            audioFileTranscript,
            audioIsPlayingOrPaused,
            chatHistory,
            clipboard,
            currentSong,
            error,
            filteredChatHistory,
            handleChangeModel,
            handleClipboardClick,
            handleCopyText,
            handleDeleteClipboard,
            handleFileUpload,
            handleFileUploadAudio,
            handleListen,
            handleListenVAD,
            handleNewChat,
            handleRegenerate,
            handleSendMessage,
            handleSendMessageAudio,
            handleSendMessageRag,
            icon,
            getAudioFileSize,
            getRagFileSize,
            getListenButtonValue,
            getMarkdown,
            audioSpeed,
            model,
            modelList,
            microPhoneOn,
            microPhoneVADOn,
            musicInfo,
            notice,
            prompt,
            ragFileUploaded,
            showMenu,
            sliderSpeed,
            sliderTemperature,
            songIndex,
            speak,
            temperature,
            ttsHost,
            uploadedFilename,
            waiting,
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
