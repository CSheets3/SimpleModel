import tensorflowjs as tfjs

print("test1")

const model = tf.sequential();

model.add(tf.layers.dense({units: 1, inputShape:[1]})); // input layer
model.add(tf.layers.dense({units:64, inputShape:[1]})); // hidden layer
model.add(tf.layers.dense({units:1, inputShape:[64]})); // output layer

// compile this bad boy
model.compile({loss: "meanSquaredError", optimizer: "sgd"});

// making the data
const xs = tf.tensor2d([1,2,3,4,5], [5,1]);
const ys = tf.tensor2d([2,4,6,8,10], [5,1]);

// running the model 1000 times (you can use any number)
model.fit(xs,ys, {epochs: 1000, shuffle: true});

// model makes a prediction
//usually comes up short by .1
model.predict(tf.tensor2d([12], [1,1])).dataSync();
