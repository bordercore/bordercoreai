import axios from "axios";

/**
 * Use axios to perform an HTTP GET call.
 * @param {string} url The url to request.
 * @param {string} callback An optional callback function.
 * @param {string} errorMsg The message to display on error.
 */
export function doGet(url, callback, errorMsg = "", responseType = "json") {
    axios.get(url, {responseType: responseType})
        .then((response) => {
            if (response.data.status && response.data.status !== "OK") {
                EventBus.$emit(
                    "toast",
                    {
                        title: "Error!",
                        body: errorMsg,
                        variant: "danger",
                        autoHide: false,
                    },
                );
                console.log(errorMsg);
            } else {
                return callback(response);
            }
        })
        .catch((error) => {
            EventBus.$emit(
                "toast",
                {
                    title: "Error!",
                    body: `${errorMsg}: ${error.message}`,
                    variant: "danger",
                    autoHide: false,
                },
            );
            console.error(error);
        });
}

/**
 * Use axios to perform an HTTP POST call.
 * @param {string} url The url to request.
 * @param {string} params The parameters for the POST body.
 * @param {string} callback An optional callback function.
 * @param {string} successMsg The message to display on success.
 * @param {string} errorMsg The message to display on error.
 */
export function doPost(url, params, callback, successMsg = "", errorCallback = () => {}) {
    const bodyFormData = new URLSearchParams();

    for (const [key, value] of Object.entries(params)) {
        bodyFormData.append(key, value);
    }

    axios(url, {
        method: "POST",
        data: bodyFormData,
    }).then((response) => {
        if (response.data.status && response.data.status !== "OK") {
            EventBus.$emit(
                "toast",
                {
                    title: "Error",
                    body: response.data.message,
                    variant: response.data.status === "Warning" ? "warning" : "danger",
                    autoHide: false,
                },
            );
            errorCallback();
            console.log("Error: ", response.data.message);
        } else {
            const body = response.data.message ? response.data.message : successMsg;
            if (body) {
                EventBus.$emit(
                    "toast",
                    {
                        title: "Success",
                        body: response.data.message ? response.data.message : successMsg,
                        variant: "info",
                    },
                );
            }
            callback(response);
        }
    })
        .catch((error) => {
            EventBus.$emit(
                "toast",
                {
                    title: "Error",
                    body: error.message,
                    variant: "danger",
                    autoHide: false,
                },
            );
            errorCallback();
            console.error(error);
        });
}

/*
  Mostly copied from https://github.com/linto-ai/WebVoiceSDK
*/

export function encodeWAV(
    samples,
    format = 3,
    sampleRate = 16000,
    numChannels = 1,
    bitDepth = 32,
) {
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    const view = new DataView(buffer);
    /* RIFF identifier */
    writeString(view, 0, "RIFF");
    /* RIFF chunk length */
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    /* RIFF type */
    writeString(view, 8, "WAVE");
    /* format chunk identifier */
    writeString(view, 12, "fmt ");
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, format, true);
    /* channel count */
    view.setUint16(22, numChannels, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * blockAlign, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, blockAlign, true);
    /* bits per sample */
    view.setUint16(34, bitDepth, true);
    /* data chunk identifier */
    writeString(view, 36, "data");
    /* data chunk length */
    view.setUint32(40, samples.length * bytesPerSample, true);
    if (format === 1) {
        // Raw PCM
        floatTo16BitPCM(view, 44, samples);
    } else {
        writeFloat32(view, 44, samples);
    }
    return buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function writeFloat32(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 4) {
        output.setFloat32(offset, input[i], true);
    }
}

/**
 * Trigger an animate.css animation on demand via JavaScript.
 * @param {object} node the DOM element to animate
 * @param {string} animation the animaton to use
 * @param {string} prefix the class prefix
 */
export const animateCSS = (node, animation, prefix = "animate__") =>
// We create a Promise and return it
new Promise((resolve, reject) => {
    const animationName = `${prefix}${animation}`;
    node.classList.add(`${prefix}animated`, animationName);

    // When the animation ends, we clean the classes and resolve the Promise
    function handleAnimationEnd(event) {
        event.stopPropagation();
        node.classList.remove(`${prefix}animated`, animationName);
        resolve("Animation ended");
    }

    node.addEventListener("animationend", handleAnimationEnd, {once: true});
});
