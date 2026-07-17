const CHARS = ["'", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ä", "Ö",
    "Ü", "’"];

const SCALE_DOWN = 8;
const WIDTH = 256;
const HEIGHT = 48;
const NUM_CHARS = CHARS.length + 1;  // including ctc blank
const NUM_TIMESTEPS = WIDTH / SCALE_DOWN;

let SESSION = null;


function tensor_from_img_element(img_element) {
    const canvas = document.createElement("canvas");
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    const ctx = canvas.getContext("2d");

    ctx.drawImage(img_element, 0, 0, WIDTH, HEIGHT);

    const img_data = ctx.getImageData(0, 0, WIDTH, HEIGHT).data;

    const arr = new Float32Array(WIDTH * HEIGHT);
    for (let i = 0; i < arr.length; i++) {
        const r = img_data[i * 4];
        const g = img_data[i * 4 + 1];
        const b = img_data[i * 4 + 2];
        const gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
        arr[i] = gray / 255.0 - 0.5;
    }

    return arr
}

function plot(chart_element, predictions) {

    let predictions_2d = []
    for (var t = 0; t < NUM_TIMESTEPS; t++) {
        var curr_timestep = [];
        for (var c = 0; c < NUM_CHARS; c++) {
            var prob = predictions[t * NUM_CHARS + c]
            curr_timestep.push(prob)
        }
        predictions_2d.push(curr_timestep);
    }

    var data = [
        {
            y: ["_"].concat(CHARS),
            z: predictions_2d,
            type: "heatmap",
            transpose: true,
            colorscale: 'Viridis',
        }
    ];

    var layout = {
        title: {
            text: "<b>Raw neural network output</b>"
        },
        xaxis: {
            title: {
                text: "Time-step"
            }
        },
        yaxis: {
            title: {
                text: "Character"
            }
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
    }

    const config = {
        responsive: true,
        displayModeBar: false // Removes the floating camera/zoom utility bar entirely
    };

    Plotly.newPlot(chart_element, data, layout, config);
}

function decode(predictions) {
    // max prediction per timestep

    // TODO: load from model
    let char_indices = [];
    let prob = 1;
    for (let t = 0; t < NUM_TIMESTEPS; t++) {
        let best_prob = Number.NEGATIVE_INFINITY;
        let best_char_idx = 0;
        for (let c = 0; c < NUM_CHARS; c++) {
            if (predictions[t * NUM_CHARS + c] > best_prob) {
                best_prob = predictions[t * NUM_CHARS + c];
                best_char_idx = c;
            }
        }
        prob *= best_prob;
        char_indices.push(best_char_idx);
    }

    // ctc best path
    let predicted_text = ""
    let best_path = ""
    let prev = -1;
    for (let t = 0; t < NUM_TIMESTEPS; t++) {
        best_path += char_indices[t] == 0 ? "_" : CHARS[char_indices[t] - 1];
        if (char_indices[t] != 0 && char_indices[t] != prev) {
            predicted_text += CHARS[char_indices[t] - 1];
        }

        prev = char_indices[t];
    }

    return [predicted_text, prob, best_path]
}

async function infer(img_element, chart_element) {
    clear_inference_output();
    write_inference_output("Processed image", img_element);

    if (SESSION == null) {
        write_inference_output("Error", "Model not loaded?");
        return;
    }
    let start = performance.now();
    const float32Data = tensor_from_img_element(document.getElementById(img_element), 256, 48);
    let end = performance.now();
    const pre_process_time = end - start;

    start = performance.now();
    const feeds = {input: new ort.Tensor("float32", float32Data, [1, 1, HEIGHT, WIDTH])};
    const results = await SESSION.run(feeds);
    const predictions = results[SESSION.outputNames[0]].data;
    end = performance.now();
    const infer_time = end - start;

    start = performance.now();
    [predicted_text, prob, best_path] = decode(predictions)
    end = performance.now();
    const decode_time = end - start;

    // log outputs
    write_inference_output("Decoded text", '"' + predicted_text + '"');
    write_inference_output("Best path (argmax)", '"' + best_path + '"');
    write_inference_output("Probability", prob.toFixed(3));
    write_inference_output("Time preprocessing", pre_process_time.toFixed(1) + "ms");
    write_inference_output("Time inference", infer_time.toFixed(1) + "ms");
    write_inference_output("Time decoding", decode_time.toFixed(1) + "ms");
    plot(chart_element, predictions);
}

async function init(onnx_file) {
    if (SESSION != null) {
        return;
    }
    try {
        const start = performance.now();

        write_model_loading_output("⏳ Loading model: " + onnx_file);
        SESSION = await ort.InferenceSession.create(onnx_file, {executionProviders: ["wasm"]});
        const end = performance.now();

        write_model_loading_output("✅ Model loaded: " + (end - start).toFixed(1) + "ms");

    } catch (error) {
        write_model_loading_output("❌ Error: " + error.message);
        console.error(error);
    }
}

