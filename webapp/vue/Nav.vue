<template>
    <nav class="navbar navbar-expand-lg navbar-dark py-0">
        <ul class="navbar-nav mr-auto">
            <li
                v-for="navItem in navItems"
                :key="navItem.link"
                class="nav-item"
                :class="{'active': navItem.label === active}"
            >
                <span
                    class="nav-link"
                    href="#"
                    @click="switchMode(navItem.label)"
                >{{ navItem.label }}</span>
            </li>
        </ul>
    </nav>
</template>

<script>

    export default {
        name: "Nav",
        props: {
            activeInitial: {
                type: String,
                default: "Chat",
            },
        },
        emits: ["mode"],
        setup(props, ctx) {
            const navItems = [
                {
                    "label": "Chat",
                    "link": "/",
                },
                {
                    "label": "RAG",
                    "link": "/rag",
                },
                {
                    "label": "Audio",
                    "link": "/audio",
                },
                {
                    "label": "Vision",
                    "link": "/vision",
                },
            ];

            const active = ref(props.activeInitial);

            function switchMode(mode) {
                active.value = mode;
                ctx.emit("mode", mode);
            };

            return {
                active,
                navItems,
                switchMode,
            };
        },
    };

</script>
