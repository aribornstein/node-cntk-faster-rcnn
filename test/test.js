const path = require('path');
const m = require('../');


const modelFileLocation = path.join(__dirname, 'm.model');
const imagesDir = path.join(__dirname, 'test_imgs');

model = new m.CNTKFRCNNModel({cntkModelPath : modelFileLocation,
    verbose : true});
model.evaluateDirectory(imagesDir, (err, res) => {
    if (err) {
        console.info(err)
        return;
    }
    console.info('%j', res);
});