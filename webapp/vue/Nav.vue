<template>
    <nav class="navbar navbar-expand-lg navbar-dark py-0">
        <ul class="navbar-nav mr-auto">
            <li
                v-for="navItem in navItems"
                :key="navItem.link"
                class="nav-item"
                :class="{'active': navItem.label === active}"
            >
                <a
                    class="nav-link"
                   :href="navItem.link"
                >{{ navItem.label }}</a>
            </li>
        </ul>
    </nav>
</template>

<script>

    export default {
        name: "Nav",
        props: {
            active: {
                type: String,
                default: "Chat",
            },
        },
        setup(props) {
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

            return {
                navItems,
            };
        },
    };

</script>
