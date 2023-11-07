<!-- ./src/components/Slider.vue -->
<!-- https://blog.openreplay.com/create-a-custom-range-slider-with-vue/ -->
<script setup>
    import {ref, watchEffect} from "vue";

    // define component props for the slider component
    const {min, max, step, modelValue} = defineProps({
        min: {
            type: Number,
            default: 0,
        },
        max: {
            type: Number,
            default: 100,
        },
        step: {
            type: Number,
            default: 1,
        },
        modelValue: {
            type: Number,
            default: 50,
        },
        showInput: {
            type: Boolean,
            default: true,
        },
    });

    // define emits for the slider component
    const emit = defineEmits(["update:modelValue"]);

    // define refs for the slider component
    const sliderValue = ref(modelValue);
    const slider = ref(null);

    // function to get the progress of the slider
    const getProgress = (value, min, max) => {
        return ((value - min) / (max - min)) * 100;
    };

    // function to set the css variable for the progress
    const setCSSProgress = (progress) => {
        slider.value.style.setProperty("--ProgressPercent", `${progress}%`);
    };

    // watchEffect to update the css variable when the slider value changes
    watchEffect(() => {
        if (slider.value) {
            // emit the updated value to the parent component
            emit("update:modelValue", sliderValue.value);

            // update the slider progress
            const progress = getProgress(
                sliderValue.value,
                slider.value.min,
                slider.value.max,
            );

            // define extrawidth to ensure that the end of progress is always under the slider thumb.
            const extraWidth = (100 - progress) / 10;

            // set the css variable
            setCSSProgress(progress + extraWidth);
        }
    });
</script>
<template>
    <div class="custom-slider">
        <input
            ref="slider"
            :value="sliderValue"
            type="range"
            :min="min"
            :max="max"
            :step="step"
            class="slider"
            @input="({ target }) => (sliderValue = parseFloat(target.value))"
        >
        <input
            v-if="showInput"
            :value="sliderValue"
            type="number"
            :min="min"
            :max="max"
            :step="step"
            class="input ms-3"
            @input="({ target }) => (sliderValue = parseFloat(target.value))"
        >
        <span v-else class="ms-2">
            <strong>{{ modelValue }}</strong>
        </span>
    </div>
</template>
<style scoped>
.custom-slider {
  --trackHeight: 0.5rem;
  --thumbRadius: 1rem;
}

/* style the input element with type "range" */
.custom-slider input[type="range"] {
  position: relative;
  appearance: none;
  /* pointer-events: none; */
  border-radius: 999px;
  z-index: 0;
}

/* ::before element to replace the slider track */
.custom-slider input[type="range"]::before {
  content: "";
  position: absolute;
  width: var(--ProgressPercent, 100%);
  height: 100%;
  background: #00865a;
  /* z-index: -1; */
  pointer-events: none;
  border-radius: 999px;
}

/* `::-webkit-slider-runnable-track` targets the track (background) of a range slider in chrome and safari browsers. */
.custom-slider input[type="range"]::-webkit-slider-runnable-track {
  appearance: none;
  background: #005a3c;
  height: var(--trackHeight);
  border-radius: 999px;
}

/* `::-moz-range-track` targets the track (background) of a range slider in Mozilla Firefox. */
.custom-slider input[type="range"]::-moz-range-track {
  appearance: none;
  background: #005a3c;
  height: var(--trackHeight);
  border-radius: 999px;
}

.custom-slider input[type="range"]::-webkit-slider-thumb {
  position: relative;
  /* top: 50%;
  transform: translate(0, -50%);
  */
  width: var(--thumbRadius);
  height: var(--thumbRadius);
  margin-top: calc((var(--trackHeight) - var(--thumbRadius)) / 2);
  background: #00bd7e;
  border-radius: 999px;
  pointer-events: all;
  appearance: none;
  z-index: 1;
}
</style>
