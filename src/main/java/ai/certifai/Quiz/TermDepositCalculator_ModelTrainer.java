package ai.certifai.Quiz;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class TermDepositCalculator_ModelTrainer {

    private static final int seed = 1234;
    private static final int epoch = 5;
    private static final int batchSize = 500;

    public static void main(String[] args) throws Exception {

        FileSplit fileSplit = new FileSplit(new ClassPathResource("/Quiz/Question1/train.csv").getFile());
        CSVRecordReader trainRecordReader = new CSVRecordReader(1,',');
        trainRecordReader.initialize(fileSplit);

        Schema trainSchema = new Schema.Builder()
                .addColumnsInteger("ID", "age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"))
                .addColumnCategorical("marital",
                        Arrays.asList("divorced", "married", "single"))
                .addColumnCategorical("education",
                        Arrays.asList("primary", "secondary", "tertiary", "unknown"))
                .addColumnCategorical("default",
                        Arrays.asList("yes", "no"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing",
                        Arrays.asList("yes", "no"))
                .addColumnCategorical("loan",
                        Arrays.asList("yes", "no"))
                .addColumnCategorical("contact",
                        Arrays.asList("cellular", "telephone", "unknown"))
                .addColumnsInteger("day")
                .addColumnCategorical("month",
                        Arrays.asList("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome",
                        Arrays.asList("unknown", "failure", "success", "other"))
                .addColumnCategorical("subscribed",
                        Arrays.asList("yes", "no"))
                .build();

        TransformProcess trainTransform = new TransformProcess.Builder(trainSchema)
                .categoricalToInteger("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "subscribed")
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> trainRawData = new ArrayList<>();
        while (trainRecordReader.hasNext()) {
            trainRawData.add(trainRecordReader.next());
        }

        List<List<Writable>> trainTransformedData = LocalTransformExecutor.execute(trainRawData, trainTransform);

        System.out.println(trainTransform.getFinalSchema());
        System.out.println(trainRawData.size());
        System.out.println(trainTransformedData.size());

        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(trainTransformedData);
        RecordReaderDataSetIterator recordReaderDataSetIterator = new RecordReaderDataSetIterator(collectionRecordReader, trainTransformedData.size(), 17, 2);
        DataSet trainFinalData = recordReaderDataSetIterator.next();
        SplitTestAndTrain dataSplit = trainFinalData.splitTestAndTrain(0.8);
        DataSet trainSplit = dataSplit.getTrain();
        DataSet validationSplit = dataSplit.getTest();
        trainSplit.setLabelNames(
                Arrays.asList("0", "1"));
        validationSplit.setLabelNames(
                Arrays.asList("0", "1"));

        NormalizerMinMaxScaler normalizerMinMaxScaler = new NormalizerMinMaxScaler();
        normalizerMinMaxScaler.fit(trainSplit);
        normalizerMinMaxScaler.transform(trainSplit);
        normalizerMinMaxScaler.transform(validationSplit);

        ViewIterator trainIterator = new ViewIterator(trainSplit, batchSize);
        ViewIterator validationIterator = new ViewIterator(validationSplit, batchSize);

        HashMap<Integer, Double> scheduler = new HashMap<>();
        scheduler.put(0, 1e-3);
        scheduler.put(4, 1e-4);
        scheduler.put(5, 1e-5);

        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, scheduler)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(trainIterator.inputColumns())
                        .nOut(200)
                        .build())
                .layer(new BatchNormalization())
                .layer(1,new DenseLayer.Builder()
                        .nOut(400)
                        .build())
                .layer(new BatchNormalization())
                .layer(2,new DenseLayer.Builder()
                        .nOut(600)
                        .build())
                .layer(new BatchNormalization())
                .layer(3,new DenseLayer.Builder()
                        .nOut(400)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nOut(trainIterator.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork trainModel = new MultiLayerNetwork(multiLayerConfiguration);
        trainModel.init();
        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        trainModel.setListeners(new StatsListener(storage), new ScoreIterationListener(100));

        ArrayList<Double> trainLoss = new ArrayList<>();
        ArrayList<Double> validationLoss = new ArrayList<>();
        DataSetLossCalculator trainLossCalculator = new DataSetLossCalculator(trainIterator, true);
        DataSetLossCalculator validationLossCalculator = new DataSetLossCalculator(validationIterator, true);
        for (int i = 0; i < epoch; i++) {
            trainModel.fit(trainIterator);
            trainLoss.add(trainLossCalculator.calculateScore(trainModel));
            validationLoss.add(validationLossCalculator.calculateScore(trainModel));
        }

        Evaluation trainEvaluation = trainModel.evaluate(trainIterator);
        Evaluation validationEvaluation = trainModel.evaluate(validationIterator);
        System.out.println(trainEvaluation.stats());
        System.out.println(validationEvaluation.stats());

        ModelSerializer.writeModel(trainModel, "C:/Users/Nisyhaal/Documents/Deep_Learning_With_Computer_Vision/Programming/JavaProject/Model/TermDeposits.zip", true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(normalizerMinMaxScaler, "C:/Users/Nisyhaal/Documents/Deep_Learning_With_Computer_Vision/Programming/JavaProject/Model/Normalizer.zip");

        Nd4j.getEnvironment().allowHelpers(false);
        List<List<Writable>> validationCollection = RecordConverter.toRecords(validationSplit);
        INDArray validationArray = RecordConverter.toMatrix(DataType.FLOAT, validationCollection);
        INDArray validationFeatures = validationArray.getColumns(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12, 13, 14, 15, 16);

        List<String> prediction = trainModel.predict(validationSplit);
        INDArray output = trainModel.output(validationFeatures);

        for (int i = 0; i < 20; i++) {
            System.out.println("Prediction:" + prediction.get(i) + "; Output: " + output.getRow(i));

        }

    }

    public static Schema getTrainSchema() {

        return new Schema.Builder()
                .addColumnsInteger("ID", "age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"))
                .addColumnCategorical("marital",
                        Arrays.asList("divorced", "married", "single"))
                .addColumnCategorical("education",
                        Arrays.asList("primary", "secondary", "tertiary", "unknown"))
                .addColumnCategorical("default",
                        Arrays.asList("yes", "no"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing",
                        Arrays.asList("yes", "no"))
                .addColumnCategorical("loan",
                        Arrays.asList("yes", "no"))
                .addColumnCategorical("contact",
                        Arrays.asList("cellular", "telephone", "unknown"))
                .addColumnsInteger("day")
                .addColumnCategorical("month",
                        Arrays.asList("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome",
                        Arrays.asList("unknown", "failure", "success", "other"))
                .addColumnCategorical("subscribed",
                        Arrays.asList("yes", "no"))
                .build();
    }
}