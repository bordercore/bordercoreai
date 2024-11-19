<template>
    <div class="switches">
        <hr>
        <div class="text-nowrap d-flex w-100 mt-3 me-5 mb-2">
            <o-switch v-model="enableSensor" @click="handleEnableSensor">
                Enable Sensor
            </o-switch>
            <span class="d-flex align-items-center ms-5 text-info">{{ connectionStatus }}</span>
        </div>
    </div>
</template>

<script>
export default {
    name: "StreamMessages",
    props: {
        sensorUri: {
            type: String,
            default: "",
        },
    },
    emits: ["sensor-data"],
    setup(props, ctx) {
        const connectionStatus = ref(null);
        const debug = ref(false);
        const enableSensor = ref(false);
        let eventSource;
        const messages = ref([]);

        function initEventSource() {
            eventSource = new EventSource(props.sensorUri);
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                ctx.emit("sensor-data", data);
                messages.value.unshift(data); // Add new messages at the start

                // Optional: Limit number of displayed messages
                if (messages.value.length > 100) {
                    messages.value.pop();
                }

                if (debug.value) {
                    console.log(data);
                }
            };

            eventSource.onopen = () => {
                connectionStatus.value = "Connected";
            };

            eventSource.onerror = (error) => {
                if (eventSource.readyState === EventSource.CLOSED) {
                    connectionStatus.value = "";
                } else {
                    connectionStatus.value = "Error";
                    console.error("SSE error:", error);
                }
            };
        };

        function closeConnection() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
                connectionStatus.value = "";
            }
        };

        function handleEnableSensor() {
            if (enableSensor.value) {
                closeConnection();
            } else {
                initEventSource();
            }
        }

        const isConnected = computed(() => {
            return connectionStatus.value === "connected";
        });

        onBeforeUnmount(() => {
            closeConnection();
        });

        return {
            connectionStatus,
            debug,
            enableSensor,
            eventSource,
            handleEnableSensor,
            isConnected,
            messages,
        };
    },
};
</script>

<style scoped>
.stream-container {
  padding: 1rem;
}

.status-bar {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border-radius: 4px;
}

.status-bar.connected {
  background-color: #e6ffe6;
  color: #006400;
}

.status-bar.disconnected {
  background-color: #ffe6e6;
  color: #640000;
}

.status-bar.error {
  background-color: #fff3e6;
  color: #804000;
}

.messages {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #eee;
  padding: 1rem;
  margin-bottom: 1rem;
}

.message {
  padding: 0.5rem;
  border-bottom: 1px solid #eee;
}

button {
  padding: 0.5rem 1rem;
  border-radius: 4px;
  border: 1px solid #ccc;
  cursor: pointer;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
