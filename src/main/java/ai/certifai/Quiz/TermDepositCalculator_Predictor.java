package ai.certifai.Quiz;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;



import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TermDepositCalculator_Predictor {


    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:/Users/Nisyhaal/Documents/Deep_Learning_With_Computer_Vision/Programming/JavaProject/Model/TermDeposits.zip", true);

        File testFilePath = new ClassPathResource("/Quiz/Question1/test.csv").getFile();
        CSVRecordReader testCsvRR = new CSVRecordReader(1, ',');
        testCsvRR.initialize(new FileSplit(testFilePath));

        Schema testSchema = getTestSchema();

        TransformProcess testTP = new TransformProcess.Builder(testSchema)
                .categoricalToInteger("job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome")
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> oriData = new ArrayList<>();

        while (testCsvRR.hasNext()) {
            oriData.add(testCsvRR.next());
        }
        testCsvRR.reset();

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(oriData, testTP);
        System.out.println(transformedData.size());

        Nd4j.getEnvironment().allowHelpers(false);

        INDArray transformedNDArray = RecordConverter.toMatrix(DataType.FLOAT, transformedData);

        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        NormalizerMinMaxScaler scaler = normalizerSerializer.restore("C:/Users/Nisyhaal/Documents/Deep_Learning_With_Computer_Vision/Programming/JavaProject/Model/Normalizer.zip");
        scaler.transform(transformedNDArray);

        INDArray output = model.output(transformedNDArray);
        System.out.println(output);

        List<List<Writable>> outputCollections = RecordConverter.toRecords(output);

        FileWriter fileWriter = new FileWriter("output.txt");

        for (int i = 0; i < output.size(0); i++) {

            if (output.getRow(i).getDouble(0) > output.getRow(i).getDouble(1)) {
                fileWriter.write("yes\n");
            } else {
                fileWriter.write("no\n");
            }
        }
        fileWriter.close();


    }

    static Schema getTestSchema() {

        return new Schema.Builder()
                .addColumnsInteger("ID", "age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                                "retired", "self-employed", "services", "student", "technician",
                                "unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("unknown", "secondary", "tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing", Arrays.asList("no", "yes"))
                .addColumnCategorical("loan", Arrays.asList("no", "yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .build();
    }
}