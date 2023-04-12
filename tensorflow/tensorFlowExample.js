import * as tf from '@tensorflow/tfjs';
import qna  from  '@tensorflow-models/qna'
// // Define a model for linear regression.
// const model = tf.sequential();
// model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// // Prepare the model for training: Specify the loss and the optimizer.
// model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// // Generate some synthetic data for training.
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// // Train the model using the data.
// model.fit(xs, ys).then(() => {
//   // Use the model to do inference on a data point the model hasn't seen before:
//   model.predict(tf.tensor2d([5], [1, 1])).print();
// });




const passage = `Facebook is an online social media and social networking service owned by American technology giant Meta Platforms. Created in 2004 by Mark Zuckerberg with fellow Harvard College students and roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes, its name derives from the face book directories often given to American university students. Membership was initially limited to only Harvard students, gradually expanding to other North American universities and, since 2006, anyone over 13 years old. As of December 2022, Facebook claimed 2.96 billion monthly active users,[6] and ranked third worldwide among the most visited websites.[7] It was the most downloaded mobile app of the 2010s.[8]

Facebook can be accessed from devices with Internet connectivity, such as personal computers, tablets and smartphones. After registering, users can create a profile revealing information about themselves. They can post text, photos and multimedia which are shared with any other users who have agreed to be their "friend" or, with different privacy settings, publicly. Users can also communicate directly with each other with Messenger, join common-interest groups, and receive notifications on the activities of their Facebook friends and the pages they follow.

The subject of numerous controversies, Facebook has often been criticized over issues such as user privacy (as with the Cambridge Analytica data scandal), political manipulation (as with the 2016 U.S. elections) and mass surveillance.[9] Posts originating from the Facebook page of Breitbart News, a media organization previously affiliated with Cambridge Analytica,[10] are currently among the most widely shared political content on Facebook.[11][12][13][14][15] Facebook has also been subject to criticism over psychological effects such as addiction and low self-esteem, and various controversies over content such as fake news, conspiracy theories, copyright infringement, and hate speech.[16] Commentators have accused Facebook of willingly facilitating the spread of such content,[17][18][19][20][21][22] as well as exaggerating its number of users to appeal to advertisers.[23]
`
const question = "Who created facebook?"
const model = await qna.load();
const answers = await model.findAnswers(question, passage);
console.log(answers);