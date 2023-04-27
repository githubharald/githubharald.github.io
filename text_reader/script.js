// draw empty canvas
const canvas = document.getElementById("canvas");
context = canvas.getContext('2d');
context.fillStyle = "gray";
context.fillRect(0, 0, 256, 48);


// react on uploaded image
document.getElementById('img_loader').addEventListener('change', set_user_img, false);


// load model
var sess = new onnx.InferenceSession();
output_message("LOADING MODEL");
var loadingModelPromise = sess.loadModel("text_reader/model.onnx");
loadingModelPromise.then(() => {output_message("MODEL READY");});



// render static image in canvas
function set_sample_img(img_name) {
  var image = new Image();
  image.onload = function () {
    context.drawImage(image, 0, 0)
  };
  image.src = img_name;
}


// render user image in canvas
function set_user_img(e) {
  var reader = new FileReader();
  reader.onload = function (event) {
    var image = new Image();
    image.onload = function () {
      var f = Math.min(canvas.width / image.width, canvas.height / image.height);
      context.fillStyle = "white";
      context.fillRect(0, 0, canvas.width, canvas.height);

      var w = image.width * f;
      var h = image.height * f;
      var tx = (canvas.width - w) / 2;
      var ty = (canvas.height - h) / 2;
      context.drawImage(image, tx, ty, w, h);
    }
    image.src = event.target.result;
  }
  reader.readAsDataURL(e.target.files[0]);
}

function output_message(text){
  document.getElementById("output").innerText = text;
}


// infer text
async function infer() {
  document.getElementById("output").innerText = "RUNNING INFERENCE";

  // get selected image
  const img_data = context.getImageData(0, 0, canvas.width, canvas.height);
  const input = new onnx.Tensor(new Float32Array(img_data.data), "float32");

  // run inference
  const outputMap = await sess.run([input]);
  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data;

  // extract best label per time-step
  var num_chars = chars.length + 1;
  var num_timesteps = 32; // TODO: load from model
  var labels = [];
  for (var t = 0; t < num_timesteps; t++) {
    var best_prob = Number.NEGATIVE_INFINITY;
    var best_label = 0;
    for (var c = 0; c < num_chars; c++) {
      if (predictions[t * num_chars + c] > best_prob) {
        best_prob = predictions[t * num_chars + c];
        best_label = c;
      }
    }

    labels.push(best_label);
  }

  // ctc best path decoding
  var s = ""
  var prev = -1;
  for (var t = 0; t < num_timesteps; t++) {
    if (labels[t] != 0 && labels[t] != prev) {
      s += chars[labels[t] - 1];
    }

    prev = labels[t];
  }

  output_message('Read text: "'+s+'"')
}

