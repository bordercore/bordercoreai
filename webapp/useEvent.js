import {onMounted, onBeforeUnmount} from "vue";

export default function useEvent(event, handler, options) {
    // Default to targeting the window
    let {
        target = window,
        ...listenerOptions
    } = options;

    // If "id" is an option, assume it represents and element to be targetted.
    //  Resolve the target in onMounted() and onBeforeUnmount() so that Vue
    //  can render the DOM first, if necessary.
    onMounted(() => {
        if (options.id) {
            target = document.getElementById(options.id);
        }
        target.addEventListener(event, handler, listenerOptions);
    });

    onBeforeUnmount(() => {
        if (options.id) {
            target = document.getElementById(options.id);
        }
        target.removeEventListener(event, handler, listenerOptions);
    });
};
