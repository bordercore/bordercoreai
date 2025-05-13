<template>
    <div class="row ms-2">
        <div
            class="control col-auto rounded-2 mb-2 p-1 ps-2"
            data-bs-toggle="collapse"
            data-bs-target="#expandableText"
            :aria-expanded="showText.toString()"
            aria-controls="expandableText"
            @click="handleExpandText"
        >
            <font-awesome-icon
                icon="gear"
                class="gear-icon me-2"
                :class="{spin: isSpinning}"
            />
            <span class="me-2">{{ title }}</span>
            <font-awesome-icon
                icon="chevron-up"
                class="toggle-icon me-1"
            />
        </div>
        <div
            v-if="showText"
            id="expandableText"
        >
            {{ text }}
        </div>
        <hr class="mt-3">
    </div>
</template>

<script>
    import {FontAwesomeIcon} from "@fortawesome/vue-fontawesome";

    export default {
        name: "ThinkingMessage",
        components: {
            FontAwesomeIcon,
        },
        props: {
            text: {
                type: String,
                default: "",
            },
        },
        setup(props, ctx) {
            const isSpinning = ref(false);
            const showText = ref(false);
            const title = ref("Thinking...");

            function handleExpandText() {
                showText.value = !showText.value;
            }

            function setTitle(newTitle) {
                title.value = newTitle;
            }

            function startSpinning() {
                isSpinning.value = true;
            }

            function stopSpinning() {
                isSpinning.value = false;
            }

            return {
                showText,
                handleExpandText,
                isSpinning,
                setTitle,
                startSpinning,
                stopSpinning,
                title,
            };
        },
    };
</script>

<style scoped>
    .toggle-icon {
        color: #6468a2;
        transition: transform 0.3s ease;
    }

    .control {
        background-color: #333333;
    }

    .control[aria-expanded="true"] .toggle-icon {
        transform: rotate(180deg);
    }

    .gear-icon {
        color: #6468a2;
    }

    .spin {
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        100% {
            transform: rotate(360deg);
        }
    }

</style>
