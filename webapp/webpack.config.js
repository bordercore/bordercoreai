const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const path = require("path");
const webpack = require("webpack");
const {VueLoaderPlugin} = require("vue-loader");

module.exports = (env, argv) => {

    config = {
        entry: {
            "static/css/styles": ["./static/css/styles.scss"],
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
                    test: /\.scss$/i,
                    use: [
                        MiniCssExtractPlugin.loader,
                        // Translates CSS into CommonJS
                        "css-loader",
                        // Compiles Sass to CSS
                        "sass-loader",
                    ],
                },
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
                    test: /\.(png|jpe?g|gif)$/i,
                    type: "asset/resource",
                    generator: {
                        filename: "static/img/[name][ext]"
                    }
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
