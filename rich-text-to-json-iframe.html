<!DOCTYPE html>
<html lang="en">

<head>
    <title>Rich Text to JSON</title>
    <link rel="stylesheet" href="https://cdn.quilljs.com/1.3.6/quill.snow.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
    <link rel="stylesheet" type="text/css"
        href="https://cdnjs.cloudflare.com/ajax/libs/spectrum/1.8.0/spectrum.min.css">
    <link rel="stylesheet"
        href='https://fonts.googleapis.com/css?family=Mirza|Roboto|Slabo+27px|Sofia|Inconsolata|Ubuntu|Akronim|Monoton&display=swap'>
    <style>
        html,
        body {
            background-color: white;
            margin: 0;
        }

        /* Set default font-family */
        .ql-snow .ql-tooltip::before {
            content: "Footnote";
            line-height: 26px;
            margin-right: 8px;
        }

        .ql-snow .ql-tooltip[data-mode=link]::before {
            content: "Enter footnote:";
        }

        .row {
            margin-top: 15px;
            margin-left: 0px;
            margin-bottom: 15px;
        }

        .btn-primary {
            color: #ffffff;
            background-color: #2780e3;
            border-color: #2780e3;
        }

        .btn-primary:hover {
            color: #ffffff;
            background-color: #1967be;
            border-color: #1862b5;
        }

        .btn {
            display: inline-block;
            margin-bottom: 0;
            font-weight: normal;
            text-align: center;
            vertical-align: middle;
            touch-action: manipulation;
            cursor: pointer;
            background-image: none;
            border: 1px solid transparent;
            white-space: nowrap;
            padding: 10px 18px;
            font-size: 15px;
            line-height: 1.42857143;
            border-radius: 0;
            user-select: none;
        }

        #standalone-container {
            width: 100%;
            background-color: #ffffff;
        }

        #editor-container {
            font-family: "Aref Ruqaa";
            font-size: 18px;
            height: 250px;
            width: 100%;
        }

        #toolbar-container {
            font-family: "Aref Ruqaa";
            display: flex;
            flex-wrap: wrap;
        }

        #json-container {
            max-width: 720px;
        }

        /* Set dropdown font-families */
        #toolbar-container .ql-font span[data-label="Base"]::before {
            font-family: "Aref Ruqaa";
        }

        #toolbar-container .ql-font span[data-label="Claude Monet"]::before {
            font-family: "Mirza";
        }

        #toolbar-container .ql-font span[data-label="Ukiyoe"]::before {
            font-family: "Roboto";
        }

        #toolbar-container .ql-font span[data-label="Cyber Punk"]::before {
            font-family: "Comic Sans MS";
        }

        #toolbar-container .ql-font span[data-label="Pop Art"]::before {
            font-family: "sofia";
        }

        #toolbar-container .ql-font span[data-label="Van Gogh"]::before {
            font-family: "slabo 27px";
        }

        #toolbar-container .ql-font span[data-label="Pixel Art"]::before {
            font-family: "inconsolata";
        }

        #toolbar-container .ql-font span[data-label="Rembrandt"]::before {
            font-family: "ubuntu";
        }

        #toolbar-container .ql-font span[data-label="Cubism"]::before {
            font-family: "Akronim";
        }

        #toolbar-container .ql-font span[data-label="Neon Art"]::before {
            font-family: "Monoton";
        }

        /* Set content font-families */
        .ql-font-mirza {
            font-family: "Mirza";
        }

        .ql-font-roboto {
            font-family: "Roboto";
        }

        .ql-font-cursive {
            font-family: "Comic Sans MS";
        }

        .ql-font-sofia {
            font-family: "sofia";
        }

        .ql-font-slabo {
            font-family: "slabo 27px";
        }

        .ql-font-inconsolata {
            font-family: "inconsolata";
        }

        .ql-font-ubuntu {
            font-family: "ubuntu";
        }

        .ql-font-Akronim {
            font-family: "Akronim";
        }

        .ql-font-Monoton {
            font-family: "Monoton";
        }

        .ql-color .ql-picker-options [data-value=Color-Picker] {
            background: none !important;
            width: 100% !important;
            height: 20px !important;
            text-align: center;
        }

        .ql-color .ql-picker-options [data-value=Color-Picker]:before {
            content: 'Color Picker';
        }

        .ql-color .ql-picker-options [data-value=Color-Picker]:hover {
            border-color: transparent !important;
        }
    </style>
</head>

<body>
    <div id="standalone-container">
        <div id="toolbar-container">
            <span class="ql-formats">
                <select class="ql-font">
                    <option selected>Base</option>
                    <option value="mirza">Claude Monet</option>
                    <option value="roboto">Ukiyoe</option>
                    <option value="cursive">Cyber Punk</option>
                    <option value="sofia">Pop Art</option>
                    <option value="slabo">Van Gogh</option>
                    <option value="inconsolata">Pixel Art</option>
                    <option value="ubuntu">Rembrandt</option>
                    <option value="Akronim">Cubism</option>
                    <option value="Monoton">Neon Art</option>
                </select>
                <select class="ql-size">
                    <option value="18px">Small</option>
                    <option selected>Normal</option>
                    <option value="32px">Large</option>
                    <option value="50px">Huge</option>
                </select>
            </span>
            <span class="ql-formats">
                <button class="ql-strike"></button>
            </span>
            <!-- <span class="ql-formats">
                <button class="ql-bold"></button>
                <button class="ql-italic"></button>
                <button class="ql-underline"></button>
            </span> -->
            <span class="ql-formats">
                <select class="ql-color">
                    <option value="Color-Picker"></option>
                </select>
                <!-- <select class="ql-background"></select> -->
            </span>
            <!-- <span class="ql-formats">
                <button class="ql-script" value="sub"></button>
                <button class="ql-script" value="super"></button>
            </span>
            <span class="ql-formats">
                <button class="ql-header" value="1"></button>
                <button class="ql-header" value="2"></button>
                <button class="ql-blockquote"></button>
                <button class="ql-code-block"></button>
            </span>
            <span class="ql-formats">
                <button class="ql-list" value="ordered"></button>
                <button class="ql-list" value="bullet"></button>
                <button class="ql-indent" value="-1"></button>
                <button class="ql-indent" value="+1"></button>
            </span>
            <span class="ql-formats">
                <button class="ql-direction" value="rtl"></button>
                <select class="ql-align"></select>
            </span>
            <span class="ql-formats">
                <button class="ql-link"></button>
                <button class="ql-image"></button>
                <button class="ql-video"></button>
                <button class="ql-formula"></button>
            </span> -->
            <span class="ql-formats">
                <button class="ql-link"></button>
            </span>
            <span class="ql-formats">
                <button class="ql-clean"></button>
            </span>
        </div>
        <div id="editor-container" style="height:300px;"></div>
    </div>
    <script src="https://cdn.quilljs.com/1.3.6/quill.min.js"></script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/spectrum/1.8.0/spectrum.min.js"></script>
    <script>

        // Register the customs format with Quill
        const Font = Quill.import('formats/font');
        Font.whitelist = ['mirza', 'roboto', 'sofia', 'slabo', 'inconsolata', 'ubuntu', 'cursive', 'Akronim', 'Monoton'];
        const Link = Quill.import('formats/link');
        Link.sanitize = function (url) {
            // modify url if desired
            return url;
        }
        const SizeStyle = Quill.import('attributors/style/size');
        SizeStyle.whitelist = ['10px', '18px', '20px', '32px', '50px', '60px', '64px', '70px'];
        Quill.register(SizeStyle, true);
        Quill.register(Link, true);
        Quill.register(Font, true);
        const icons = Quill.import('ui/icons');
        icons['link'] = `<svg xmlns="http://www.w3.org/2000/svg" width="17" viewBox="0 0 512 512" xml:space="preserve"><path fill="#010101" d="M276.75 1c4.51 3.23 9.2 6.04 12.97 9.77 29.7 29.45 59.15 59.14 88.85 88.6 4.98 4.93 7.13 10.37 7.12 17.32-.1 125.8-.09 251.6-.01 377.4 0 7.94-1.96 14.46-9.62 18.57-121.41.34-242.77.34-364.76.05A288.3 288.3 0 0 1 1 502c0-163.02 0-326.04.34-489.62C3.84 6.53 8.04 3.38 13 1c23.35 0 46.7 0 70.82.3 2.07.43 3.38.68 4.69.68h127.98c18.44.01 36.41.04 54.39-.03 1.7 0 3.41-.62 5.12-.95h.75M33.03 122.5v359.05h320.22V129.18h-76.18c-14.22-.01-19.8-5.68-19.8-20.09V33.31H33.02v89.19m256.29-27.36c.72.66 1.44 1.9 2.17 1.9 12.73.12 25.46.08 37.55.08L289.3 57.45v37.7z"/><path fill="#020202" d="M513 375.53c-4.68 7.99-11.52 10.51-20.21 10.25-13.15-.4-26.32-.1-39.48-.1h-5.58c5.49 8.28 10.7 15.74 15.46 23.47 6.06 9.82 1.14 21.65-9.96 24.27-6.7 1.59-12.45-.64-16.23-6.15a2608.6 2608.6 0 0 1-32.97-49.36c-3.57-5.48-3.39-11.54.17-16.98a3122.5 3122.5 0 0 1 32.39-48.56c5.22-7.65 14.67-9.35 21.95-4.45 7.63 5.12 9.6 14.26 4.5 22.33-4.75 7.54-9.8 14.9-15.11 22.95h33.64V225.19h-5.24c-19.49 0-38.97.11-58.46-.05-12.74-.1-20.12-13.15-13.84-24.14 3.12-5.46 8.14-7.71 14.18-7.73 26.15-.06 52.3-.04 78.45 0 7.1 0 12.47 3.05 16.01 9.64.33 57.44.33 114.8.33 172.62z"/><path fill="#111" d="M216.03 1.97C173.52 1.98 131 2 88.5 1.98a16 16 0 0 1-4.22-.68c43.4-.3 87.09-.3 131.24-.06.48.25.5.73.5.73z"/><path fill="#232323" d="M216.5 1.98c-.47 0-.5-.5-.5-.74C235.7 1 255.38 1 275.53 1c-1.24.33-2.94.95-4.65.95-17.98.07-35.95.04-54.39.03z"/><path fill="#040404" d="M148 321.42h153.5c14.25 0 19.96 5.71 19.96 19.97.01 19.17.03 38.33 0 57.5-.03 12.6-6.16 18.78-18.66 18.78H99.81c-12.42 0-18.75-6.34-18.76-18.73-.01-19.83-.02-39.66 0-59.5.02-11.47 6.4-17.93 17.95-18 16.17-.08 32.33-.02 49-.02m40.5 32.15h-75.16v31.84h175.7v-31.84H188.5z"/><path fill="#030303" d="m110 225.33 178.89-.03c11.98 0 19.25 9.95 15.74 21.44-2.05 6.71-7.5 10.57-15.14 10.57-63.63 0-127.25-.01-190.88-.07-12.03-.02-19.17-8.62-16.7-19.84 1.6-7.21 7.17-11.74 15.1-12.04 4.17-.16 8.33-.03 13-.03zm-24.12-36.19c-5.28-6.2-6.3-12.76-2.85-19.73 3.22-6.49 9.13-8.24 15.86-8.24 25.64.01 51.27-.06 76.91.04 13.07.04 20.66 10.44 16.33 22.08-2.25 6.06-6.63 9.76-13.08 9.8-27.97.18-55.94.2-83.9-.07-3.01-.03-6-2.36-9.27-3.88z"/></svg>`
        const quill = new Quill('#editor-container', {
            modules: {
                toolbar: {
                    container: '#toolbar-container',
                },
            },
            theme: 'snow'
        });
        var toolbar = quill.getModule('toolbar');
        $(toolbar.container).find('.ql-color').spectrum({
            preferredFormat: "rgb",
            showInput: true,
            showInitial: true,
            showPalette: true,
            showSelectionPalette: true,
            palette: [
                ["#000", "#444", "#666", "#999", "#ccc", "#eee", "#f3f3f3", "#fff"],
                ["#f00", "#f90", "#ff0", "#0f0", "#0ff", "#00f", "#90f", "#f0f"],
                ["#ea9999", "#f9cb9c", "#ffe599", "#b6d7a8", "#a2c4c9", "#9fc5e8", "#b4a7d6", "#d5a6bd"],
                ["#e06666", "#f6b26b", "#ffd966", "#93c47d", "#76a5af", "#6fa8dc", "#8e7cc3", "#c27ba0"],
                ["#c00", "#e69138", "#f1c232", "#6aa84f", "#45818e", "#3d85c6", "#674ea7", "#a64d79"],
                ["#900", "#b45f06", "#bf9000", "#38761d", "#134f5c", "#0b5394", "#351c75", "#741b47"],
                ["#600", "#783f04", "#7f6000", "#274e13", "#0c343d", "#073763", "#20124d", "#4c1130"]
            ],
            change: function (color) {
                var value = color.toHexString();
                quill.format('color', value);
            }
        });

        quill.on('text-change', () => {
            // keep qull data inside _data to communicate with Gradio
            document.body._data = quill.getContents()
        })
        function setQuillContents(content) {
            quill.setContents(content);
            document.body._data = quill.getContents();
        }
        document.body.setQuillContents = setQuillContents
    </script>
</body>

</html>