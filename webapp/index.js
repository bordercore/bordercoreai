import {doGet, doPost, encodeWAV, animateCSS, isValidURL} from "./util.js";

import axios from "axios";
import "bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import {library} from "@fortawesome/fontawesome-svg-core";
import {FontAwesomeIcon} from "@fortawesome/vue-fontawesome";
import {faBackward, faCheck, faChevronUp, faCopy, faExclamationTriangle, faFileAlt, faForward, faGear, faLink, faPaperclip, faPaste, faPlus, faRotateLeft} from "@fortawesome/free-solid-svg-icons";
library.add(faBackward, faCheck, faChevronUp, faCopy, faExclamationTriangle, faFileAlt, faForward, faGear, faLink, faPaperclip, faPaste, faPlus, faRotateLeft);
import "media-chrome";
import {Modal} from "bootstrap";
import Oruga from "@oruga-ui/oruga-next";
import "@oruga-ui/oruga-next/dist/oruga-full.css";
import "@oruga-ui/oruga-next/dist/oruga-full-vars.css";
import {computed, createApp, nextTick, onBeforeUnmount, onMounted, ref, watch} from "vue";
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
window.FontAwesomeIcon = FontAwesomeIcon;
window.computed = computed;
window.onMounted = onMounted;
window.onBeforeUnmount = onBeforeUnmount;
window.ref = ref;

import Nav from "./vue/Nav.vue";
window.MyNav = Nav;
import StreamMessages from "./vue/StreamMessages.vue";
window.StreamMessages = StreamMessages;
import ThinkingMessage from "./vue/ThinkingMessage.vue";
window.ThinkingMessage = ThinkingMessage;

const app = createApp({
    name: "ChatBot",
    delimiters: ["[[", "]]"],
    components: {
        FontAwesomeIcon,
        MyNav,
        Slider,
        StreamMessages,
        ThinkingMessage,
    },
    setup() {
        const session = JSON.parse(document.getElementById("session").textContent);
        const settings = JSON.parse(document.getElementById("settings").textContent);
        const controlValue = document.getElementById("controlValue").textContent;

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
        const isDragOver = ref(false);
        const mode = ref("Chat");
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
        let sensorDetectMode = true;
        const speak = ref(session.speak !== undefined ? session.speak : true);
        const temperature = ref(session.temperature || 0.7);
        const visionImage = ref(null);
        const waiting = ref(false);

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
        const wolframAlpha = ref(false);
        const enableThinking = ref(false);
        const url = ref("");

        const sliderSpeed = ref(null);
        const sliderTemperature = ref(null);
        const thinkingMessage = ref(null);
        let thinkingStarted = null;

        watch(thinkingMessage, (newValue, oldValue) => {
            if (newValue) {
                thinkingStarted = Date.now();
                newValue[0].startSpinning();
            }
        });

        const sensorThreshold = settings.sensor_threshold ?? 100;

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

        const inputIsDisabled = computed(() => {
            if (mode.value === "Vision" && !visionImage.value) {
                return true;
            } else if (mode.value === "RAG" && !ragFileUploaded.value) {
                return true;
            } else {
                return false;
            }
        });

        const chatEndpoint = computed(() => {
            switch (mode.value) {
            case "Audio":
                return "/audio/chat";
            case "Vision":
                return "/vision/chat";
            case "RAG":
                return "/rag/chat";
            default:
                return "/chat";
            }
        });

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
                const scrollableDiv = document.getElementById("message-container");
                scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
            });
        };

        function convertBase64ToBytes(data) {
            const base64Clean = data.audio.replace(/^data:audio\/\w+;base64,/, "");
            const byteCharacters = atob(base64Clean);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], {type: "audio/mp3"});
        };

        function getMarkdown(content) {
            return markdown.render(content);
        };

        function getModelAttribute(modelName, attribute) {
            const result = modelList.value.find((obj) => obj.model === modelName);
            return result ? result[attribute] : "";
        };

        function handleChangeModel(event) {
            const modelType = getModelAttribute(event.srcElement.value, "type");
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

        function handleImageDrop(event) {
            const image = event.dataTransfer.files[0];
            if (image.type.indexOf("image/") >= 0) {
                handleFileUploadVision(event);
            }
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

        function handleTranscribeAudio(event, url=null) {
            const modal = new Modal("#modalProcessing");
            modal.show();

            const formData = new FormData();
            let fileData;
            let endpoint;

            if (url) {
                endpoint = "audio/upload/url";
                formData.append("url", url);
            } else {
                endpoint = "audio/upload/file";
                fileData = event.target.files[0];
                if (!fileData) {
                    return;
                }
                formData.append("file", fileData);
            }

            axios.post(
                endpoint,
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }).then((response) => {
                    audioFileTranscript.value = response.data.text;
                    audioFileSize.value = audioFileTranscript.value.length;
                    if (url) {
                        uploadedFilename.value = response.data.title;
                    } else {
                        uploadedFilename.value = event.target.files[0].name;
                    }
                    modal.hide();
                    // Load the audio into the player
                    let el = document.getElementById("audioPlayer");
                    el.classList.replace("d-none", "d-flex");
                    el = document.getElementById("player");
                    if (url) {
                        fileData = convertBase64ToBytes(response.data);
                    }
                    const audioURL = URL.createObjectURL(fileData);
                    el.src = audioURL;
                });
        };

        function handleFileUploadVision(event) {
            if (event.target.files) {
                visionImage.value = event.target.files[0];
            } else {
                visionImage.value = event.dataTransfer.files[0];
            }

            const reader = new FileReader();
            reader.onload = function(event) {
                const imagePreview = document.getElementById("image-preview");
                imagePreview.src = event.target.result;
            };
            reader.readAsDataURL(visionImage.value);
        };

        function handleSongBackward(event) {
            if (songIndex.value > 0) {
                nextTick(() => {
                    animateCSS(event.currentTarget, "heartBeat");
                });
                playSong(musicInfo.value[songIndex.value - 1]);
            }
        };

        function handleSongForward(event) {
            if (songIndex.value < musicInfo.value.length - 1) {
                nextTick(() => {
                    animateCSS(event.currentTarget, "heartBeat");
                });
                playSong(musicInfo.value[songIndex.value + 1]);
            }
        };

        function handleNewChat(event) {
            chatHistory.value.length = 1;
            clipboard.value = null;
            url.value = "";
            error.value = "";
            visionImage.value = null;
        };

        function handleSendMessage(event, message=prompt.value) {
            if (mode.value === "Rag") {
                handleSendMessageRag(event);
            } else if (mode.value === "Audio") {
                handleSendMessageAudio();
            } else if (mode.value === "Vision") {
                handleSendMessageVision();
            } else {
                sendMessageToChatbot(message);
            }
        };

        function handleSendMessageRag() {
            const args = {
                "sha1sum": sha1sum.value,
            };
            sendMessageToChatbot(prompt.value, args);
        };

        function handleSendMessageAudio() {
            if (!audioFileTranscript.value) {
                return;
            }
            const args = {
                "transcript": audioFileTranscript.value,
            };
            sendMessageToChatbot(prompt.value, args);
        };

        function handleSendMessageVision(event, regenerate=false) {
            const reader = new FileReader();
            reader.onload = function(event) {
                const args = {
                    image: event.target.result,
                };
                sendMessageToChatbot(prompt.value, args, regenerate);
            };
            reader.readAsDataURL(visionImage.value);
        };

        function handleSensorData(event) {
            if (event.moving_target_energy == sensorThreshold && sensorDetectMode) {
                sensorDetectMode = false;
                handleListen();
                microPhoneOn.value = !microPhoneOn.value;
                setTimeout(function() {
                    sensorDetectMode = true;
                }, 2000);
            }
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
            if (mode.value === "Vision") {
                handleSendMessageVision(null, true);
            } else {
                sendMessageToChatbot(prompt.value, {}, true);
            }
        };

        function handleSwitchMode(modeNew) {
            mode.value = modeNew;
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

        function removeThinkingField(array) {
            return array.map((obj) => {
                // Create a new object by copying properties
                const newObj = {...obj};

                // Remove the "thinking" property if it exists
                if ("thinking" in newObj) {
                    delete newObj.thinking;
                }

                return newObj;
            });
        };

        async function sendMessageToChatbot(message, args = {}, regenerate = false) {
            setTimeout(function() {
                if (!error.value) {
                    waiting.value = true;
                }
            }, 500);

            if (
                mode.value === "Vision" &&
                    getModelAttribute(model.value, "qwen_vision") === null
            ) {
                error.value = {
                    body: "Error: you must load a vision model to use this feature.",
                    variant: "danger",
                };
                return;
            }

            if (regenerate) {
                if (!error.value) {
                    chatHistory.value.pop();
                }
            } else {
                addMessage("user", message);
            }

            error.value = "";
            const messages = removeThinkingField(addClipboardToMessages());
            prompt.value = "";

            const payload = {
                message: JSON.stringify(messages),
                model: model.value,
                audio_speed: audioSpeed.value,
                speak: speak.value,
                temperature: temperature.value,
                wolfram_alpha: wolframAlpha.value,
                enable_thinking: enableThinking.value,
                url: url.value,
                ...args,
            };

            const formData = new FormData();
            for (const key in payload) {
                if (payload.hasOwnProperty(key)) {
                    formData.append(key, payload[key]);
                }
            }

            let start = null;
            let buffer = "";
            let revealed = enableThinking.value ? false : true;

            fetch(chatEndpoint.value, {
                method: "POST",
                headers: {
                    Responsetype: "stream",
                },
                body: formData,
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Network response was not ok");
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");

                    addMessage("assistant", "");

                    return new ReadableStream({
                        start(controller) {
                            function push() {
                                reader
                                    .read()
                                    .then(({done, value}) => {
                                        if (done) {
                                            controller.close();
                                            return;
                                        }

                                        waiting.value = false;
                                        const content = decoder.decode(value, {
                                            stream: true,
                                        });

                                        buffer += content;

                                        // Handle JSON control payloads if any
                                        if (
                                            buffer.slice(0, controlValue.length) ===
                                                controlValue
                                        ) {
                                            // Do nothing here; will be handled in final .then()
                                        } else if (!revealed) {
                                            const thinkCompleteMatch = buffer.match(
                                                /^<think>(.*?)<\/think>([\s\S]*)/s,
                                            );
                                            if (thinkCompleteMatch) {
                                                const realContent = thinkCompleteMatch[2];
                                                chatHistory.value[
                                                    chatHistory.value.length - 1
                                                ].content += realContent;
                                                chatHistory.value[
                                                    chatHistory.value.length - 1
                                                ].thinking = thinkCompleteMatch[1];
                                                revealed = true;
                                                if (thinkingMessage.value) {
                                                    thinkingMessage.value[0].setTitle(`Thought for ${Math.floor((Date.now() - thinkingStarted) / 1000)} seconds`);
                                                    thinkingMessage.value[0].stopSpinning();
                                                }
                                            }
                                        } else {
                                            chatHistory.value[
                                                chatHistory.value.length - 1
                                            ].content += content;
                                        }

                                        controller.enqueue(value);
                                        push();
                                    })
                                    .catch((error) => {
                                        console.error(error);
                                        controller.error(error);
                                    });
                            }
                            push();
                        },
                    });
                })
                .then((stream) => {
                    start = Date.now();
                    return new Response(stream).text();
                })
                .then((result) => {
                    const elapsed = Date.now() - start;
                    const wordCount = result.trim().split(/\s+/).length;
                    const speed = Math.round(wordCount / (elapsed / 1000));
                    console.log(`Speed: ${speed} t/s`);

                    if (
                        buffer.slice(0, controlValue.length) === controlValue &&
                            buffer.length > controlValue.length
                    ) {
                        const jsonObject = JSON.parse(
                            buffer.slice(controlValue.length),
                        );
                        if (jsonObject?.music_info) {
                            if (jsonObject.music_info.length > 0) {
                                musicInfo.value = jsonObject.music_info;
                                playSong(musicInfo.value[0]);
                            } else {
                                chatHistory.value[
                                    chatHistory.value.length - 1
                                ].content = "No music found.";
                            }
                        }
                    }

                    doTTS(result);
                })
                .catch((exception) => {
                    error.value = {
                        body: "Error communicating with webapp.",
                        variant: "danger",
                    };
                    console.error("Error:", exception);
                });
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
                                notice.value = "";
                                handleSendMessage(null, response.data.input);
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
                    // Stop any TTS currently playing from previous responses
                    audioElement.pause();
                    audioElement.src = "";

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
                            handleSendMessage(null, response.data.input);
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

            const voice = session.tts_voice;
            const outputFile = "stream_output.wav";
            const streamingUrl = `http://${ttsHost.value}/?text=${encodeURIComponent(response)}&voice=${voice}&language=en&output_file=${outputFile}`;
            audioElement.src = streamingUrl;
            audioMotion.gradient = "steelblue";
            audioMotion.volume = 1;
            audioElement.playbackRate = audioSpeed.value;
            audioElement.play();
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

            if (isValidURL(paste)) {
                url.value = paste;
                prompt.value = "";
                if (mode.value === "Audio") {
                    handleTranscribeAudio(event, url.value);
                }
                return;
            };

            if (paste.length > 200) {
                clipboard.value = {"content": paste, "id": id + 1};
            } else {
                prompt.value += paste;
            }
        };

        async function loadOptionalModule(uuid) {
            return await import(/* webpackChunkName: "optional" */ "@optional-module");
            module.run(uuid);
        };

        async function playSong(song) {
            chatHistory.value[chatHistory.value.length - 1].content = `Playing **${song.title}** by **${song.artist}**`;
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
            if (songIndex.value < musicInfo.value.length - 1) {
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

        function getVisionFileSize() {
            return formatBytes(visionImage.value.size);
        };

        function formatBytes(bytes, decimals = 2) {
            if (!+bytes) return "0 Bytes";

            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ["b", "k", "M", "G", "T", "P", "E", "Z", "Y"];

            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return `${parseInt((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
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

            setTimeout( () => {
                document.getElementById("prompt").focus();
            });
        });

        return {
            audioFileTranscript,
            audioIsPlayingOrPaused,
            chatHistory,
            clipboard,
            currentSong,
            enableThinking,
            error,
            filteredChatHistory,
            handleChangeModel,
            handleClipboardClick,
            handleCopyText,
            handleDeleteClipboard,
            handleFileUpload,
            handleTranscribeAudio,
            handleFileUploadVision,
            handleImageDrop,
            handleSongBackward,
            handleSongForward,
            handleListen,
            handleListenVAD,
            handleNewChat,
            handleRegenerate,
            handleSendMessage,
            handleSendMessageAudio,
            handleSendMessageRag,
            handleSendMessageVision,
            handleSensorData,
            handleSwitchMode,
            icon,
            inputIsDisabled,
            isDragOver,
            getAudioFileSize,
            getRagFileSize,
            getMarkdown,
            getVisionFileSize,
            audioSpeed,
            mode,
            model,
            modelList,
            microPhoneOn,
            microPhoneVADOn,
            musicInfo,
            notice,
            prompt,
            ragFileUploaded,
            settings,
            showMenu,
            sliderSpeed,
            sliderTemperature,
            songIndex,
            speak,
            temperature,
            thinkingMessage,
            ttsHost,
            uploadedFilename,
            url,
            visionImage,
            waiting,
            wolframAlpha,
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
