let model;

function createPredictionInput() {
    const input = document.createElement('input');
    input.type = 'number';
    input.id = 'predict-input';

    document.querySelector('#predict').appendChild(input);
}

function createPredictionOutputParagraph() {
    const p = document.createElement('p');
    p.id = 'predict-output-p';

    document.querySelector('#predict').appendChild(p);
}

function createPredictButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Predict!';
  btn.id = 'predict-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    // Get the value from the input
    const valueToPredict = document.getElementById('predict-input').value;
    const parsedValue = parseInt(valueToPredict, 10);
    const prediction = model.predict(tf.tensor1d([parsedValue])).dataSync();

    // Get the <p> element and append the prediction result
    const p = document.getElementById('predict-output-p');
    p.innerHTML = `Predicted value is: ${prediction}`;
  });

  document.querySelector('#predict').appendChild(btn);
}

async function init(){
    createPredictionInput();
    createPredictionOutputParagraph();
    createPredictButton();
    model = await tf.loadLayersModel('model/model.json');
}

init();