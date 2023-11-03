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
export function doPost(url, params, callback, successMsg = "", errorMsg = "") {
    const bodyFormData = new URLSearchParams();

    for (const [key, value] of Object.entries(params)) {
        bodyFormData.append(key, value);
    }

    axios(url, {
        method: "POST",
        data: bodyFormData,
    }).then((response) => {
        if (response.data.status && response.data.status === "Warning") {
            EventBus.$emit(
                "toast",
                {
                    title: "Error",
                    body: response.data.message,
                    variant: "warning",
                    autoHide: false,
                },
            );
            console.log("Warning: ", response.data.message);
        } else if (response.data.status && response.data.status !== "OK") {
            EventBus.$emit(
                "toast",
                {
                    title: "Error",
                    body: response.data.message,
                    variant: "danger",
                    autoHide: false,
                },
            );
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
            console.error(error);
        });
}
