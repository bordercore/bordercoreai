const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const path = require("path");
const webpack = require("webpack");
const {VueLoaderPlugin} = require("vue-loader");

module.exports = (env, argv) => {

    config = {
        entry: {
            "static/js/javascript": "./index.js",
        },
        mode: "development",
        output: {
            path: path.resolve(__dirname, "."),
        },
        resolve: {
            alias: {
                vue$: "vue/dist/vue.esm-bundler.js",
            },
        },
        plugins: [
            new MiniCssExtractPlugin({
                filename: "[name].css",
            }),

            new VueLoaderPlugin(),

            // Define these to improve tree-shaking and muffle browser warnings
            new webpack.DefinePlugin({
                __VUE_OPTIONS_API__: true,
                __VUE_PROD_DEVTOOLS__: false,
            }),
        ],
        module: {
            rules: [
                {
                    test: /\.css$/,
                    use: [
                        "style-loader",
                        {
                            loader: "css-loader",
                            options: {
                                esModule: false,
                            },
                        },
                    ],
                },
               {
                   test: /\.vue$/,
                   loader: "vue-loader",
                },
            ],
        },
    };

    return config;
};
