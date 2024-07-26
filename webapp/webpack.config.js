const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const path = require("path");
const webpack = require("webpack");
const {VueLoaderPlugin} = require("vue-loader");
const fs = require("fs");

module.exports = (env, argv) => {

    config = {
        entry: {
            "static/css/styles": ["./static/css/styles.scss"],
            "static/js/javascript": "./index.js",
        },
        mode: "development",
        output: {
            path: path.resolve(__dirname, "."),
            filename: "[name].bundle.js",
            chunkFilename: "./static/js/[name].chunk.js",
        },
        resolve: {
            alias: {
                vue$: "vue/dist/vue.esm-bundler.js",
                "@optional-module": path.resolve(__dirname, "./local/optional.js"),
            },
        },
        plugins: [
            new MiniCssExtractPlugin({
                filename: "[name].css",
            }),

            // If optional.js doesn't exist, use fallback.js instead
            new webpack.NormalModuleReplacementPlugin(
                /@optional-module/,
                (resource) => {
                    const modulePath = path.resolve(__dirname,  "./local/optional.js");
                    if (!fs.existsSync(modulePath)) {
                        resource.request = path.resolve(__dirname, "fallback.js");
                    }
                }
            ),

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
