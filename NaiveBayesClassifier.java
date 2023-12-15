// I have neither given nor received unauthorized aid on this program
import java.io.*;
import java.util.*;
public class NaiveBayesClassifier {
    private static Integer cnt = 0;
    private static int spamCnt = 0;
    private static int hamCnt = 0;
    private static final Set<String> vocab = new HashSet<>();
    private static final Map<String, Integer> spamWords = new HashMap<>();
    private static final Map<String, Integer> hamWords = new HashMap<>();
    private static double priorSpam;
    private static double priorHam;
    private static String spamOrHam;
    private static int trueCnt;

    public static void main(String[] args) throws IOException {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the name of the training spam file: ");
        String trainSpamFile = scanner.nextLine();
        System.out.print("Enter the name of the training ham file: ");
        String trainHamFile = scanner.nextLine();

        System.out.print("Enter the name of the testing spam file: ");
        String testSpamFile = scanner.nextLine();
        System.out.print("Enter the name of the testing ham file: ");
        String testHamFile = scanner.nextLine();

        System.out.print("Would you like debugging info (y/n): ");
        String debug = scanner.nextLine();

        System.out.println("Training from " + trainSpamFile + " and " + trainHamFile);
        System.out.println("Testing from " + testSpamFile + " and " + testHamFile);

        classifier.parseTrain(trainSpamFile, spamWords, true);
        classifier.parseTrain(trainHamFile, hamWords, false);

        classifier.calculatePriors();

        // check if the user chose debugging info or not
        if(debug.equalsIgnoreCase("Y")){
            classifier.printDebugOutput(testSpamFile,testHamFile);
        }
        else if(debug.equalsIgnoreCase("N")){
            classifier.printNoDebugOutput(testSpamFile,testHamFile);
        }

        scanner.close();
    }

    private static void parseTrain(String filename, Map<String, Integer> specialWordCnts, boolean isSpam) throws IOException {
        Scanner scan = new Scanner(new File(filename));
        Set<String> seenWords = null; // Set to track the words seen in each email

        // loops through each line in the file
        while (scan.hasNextLine()) {
            String line = scan.nextLine().toLowerCase(); // reads the next line and converts it to lowercase

            // check if the line is the start of a new email
            if (line.startsWith("<subject>")) {
                seenWords = new HashSet<>(); //resets the set for each email
                if (isSpam) {
                    spamCnt++;
                } else {
                    hamCnt++;
                }
            }

            // If the line starts with one of these words then skip over the line
            if (line.isEmpty() || line.startsWith("<subject>") || line.startsWith("<body>")
                    || line.startsWith("</subject>") || line.startsWith("</body>")) {
                continue;
            }

            Scanner wordScan = new Scanner(line); // reads each word

            // loops through each word in the current line
            while (wordScan.hasNext()) {
                String word = wordScan.next();
                vocab.add(word);

                if (!seenWords.contains(word)) {
                    cnt = specialWordCnts.get(word);
                    seenWords.add(word); // adds the word to the set of seen words for the current email
                    if (cnt == null) {
                        specialWordCnts.put(word, 1); // if the word is not yet in the map initialize it to 1
                    } else {
                        specialWordCnts.put(word, cnt + 1); // if the word is already in the map then add 1 to cnt
                    }
                }


            }
            wordScan.close();
        }
        scan.close();
    }

    private static List <Set<String>> parseTest(String filename) throws FileNotFoundException {
        Scanner scan = new Scanner(new File(filename));
        List<Set<String>> seenWords = new ArrayList<>();
        int cntEmails = 0;
        // loops through each line in the file
        while (scan.hasNextLine()) {
            String line = scan.nextLine().toLowerCase();

            // check if the line is the start of a new email
            if (line.startsWith("<subject>")) {
                cntEmails++;
                seenWords.add(new HashSet<>()); // adds a new set to the list for the new email
            }
            // If the line starts with one of these words then skip over the line
            if (line.isEmpty() || line.startsWith("<subject>") || line.startsWith("<body>")
                    || line.startsWith("</subject>") || line.startsWith("</body>")) {
                continue;
            }

            Scanner lineScan = new Scanner(line);

            // loops through each word in the line
            while (lineScan.hasNext()) {
                String word = lineScan.next();
                // if the word is in the vocab then add the word to the set of the current email
                if(vocab.contains(word)) {
                    seenWords.get(cntEmails - 1).add(word);
                }
            }
            lineScan.close();
        }
        scan.close();

        return seenWords;
    }
    // Calculates the prior values for spam and ham
    private static void calculatePriors(){
        priorSpam = (double)spamCnt/(spamCnt + hamCnt);
        priorHam = (double) hamCnt/(spamCnt + hamCnt);
    }

    private static double calculateProb(String word, Map<String, Integer> occurrences, int totalWords){
        Integer occur = occurrences.get(word);
        // if the word is not in the map then set its count to 0
        if(occur == null){
            occur = 0;
        }
        return (occur + 1.0)/(totalWords + 2.0);
    }


    private static double CalcLikelihoodNoLog(Set<String> words, boolean isSpam){
        double LikelihoodNoLog = 1.0; // variable to store the product of the probabilities
        Map<String, Integer> wordCnts; // stores the count of each word in either the spam or ham training datasets
        int totalWordCnt;

        if(isSpam){
            wordCnts = spamWords;
            totalWordCnt = spamCnt;
        }
        else{
            wordCnts = hamWords;
            totalWordCnt = hamCnt;
        }

        // iterates through each word in the vocabulary
        for(String word : vocab){
            // check if the word is in the email
            if(words.contains(word) && isSpam){
                // if the word is in the email multiply the probability of true word given spam or ham
                LikelihoodNoLog *= calculateProb(word,wordCnts,totalWordCnt);
            }
            else {
                // if the word is not in the email then multiply the probability of false word given spam or ham and -1 to get the complement
                LikelihoodNoLog *= 1.0 - calculateProb(word,wordCnts, totalWordCnt);
            }
        }

        // multiplies the priors to the spam or ham LikelihoodNoLog value
        if(isSpam){
            LikelihoodNoLog *= priorSpam;
        }
        else{
            LikelihoodNoLog *= priorHam;
        }

        return LikelihoodNoLog;
    }

    private static double CalcLikelihoodNoPrior(Set<String> words, boolean isSpam){
        double LikelihoodNoLog = 1.0; // variable to store the product of the probabilities
        Map<String, Integer> wordCnts; // stores the count of each word in either the spam or ham training datasets
        int totalWordCnt;

        if(isSpam){
            wordCnts = spamWords;
            totalWordCnt = spamCnt;
        }
        else{
            wordCnts = hamWords;
            totalWordCnt = hamCnt;
        }

        // iterates through each word in the vocabulary
        for(String word : vocab){
            // check if the word is in the email
            if(words.contains(word)){
                // if the word is in the email multiply the probability of true word given spam or ham
                LikelihoodNoLog *= calculateProb(word,wordCnts,totalWordCnt);
            }
            else {
                // if the word is not in the email then multiply the probability of false word given spam or ham and -1 to get the complement
                LikelihoodNoLog *= 1.0 - calculateProb(word,wordCnts, totalWordCnt);
            }
        }

        return LikelihoodNoLog;
    }

    private static double CalcLikelihoodLog(Set<String> words, boolean isSpam){
        trueCnt = 0; // number of words that were true in this message
        double logLikelihood = 0; // variable to store the sum of the log probabilities
        Map<String, Integer> wordCnts; // stores the count of each word in either the spam or ham training datasets
        int totalWordCnt;

        if(isSpam){
            wordCnts = spamWords;
            totalWordCnt = spamCnt;
        }
        else{
            wordCnts = hamWords;
            totalWordCnt = hamCnt;
        }

        // iterates through each word in the vocabulary
        for(String word : vocab){
            // checks if the word is in the email,
            if(words.contains(word)){
                trueCnt++;
                // if the word is in the email the add the probability of true word given spam or ham
                logLikelihood += Math.log(calculateProb(word,wordCnts,totalWordCnt));
            }
            else {
                // if the word is not in the email then add the probability of false word given spam or ham and -1 to get the complement
                logLikelihood += Math.log(1.0 - calculateProb(word,wordCnts, totalWordCnt));
            }
        }

        // adds the priors to the spam or ham logLikelihood value
        if(isSpam){
            logLikelihood += Math.log(priorSpam);
        }
        else{
            logLikelihood += Math.log(priorHam);
        }

        return logLikelihood;
    }

    public static void printDebugOutput(String spamFile, String hamFile) throws FileNotFoundException {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();
        System.out.println("number of emails " + spamCnt + " vs " +hamCnt);
        System.out.println("entire vocab" + vocab);
        System.out.println("entire vocab size is " + vocab.size());
        System.out.println("spam words " + spamWords);
        System.out.println("ham words " + hamWords);
        System.out.println("Beginning tests.");

        List<Set<String>> words = classifier.parseTest(spamFile);
        int numSpamCorrect = 0;
        int numHamCorrect = 0;
        int totalEmails = 0;
        String rightOrWrong;
        System.out.println("Testing spam emails.");
        for(int i = 0; i < words.size(); i++) {
            System.out.println("Test email " + (i + 1));
            System.out.println("priors= " + priorSpam + " " + priorHam);
            System.out.println("likelihoods= " + classifier.CalcLikelihoodNoPrior(words.get(i),true) + " " + classifier.CalcLikelihoodNoPrior(words.get(i),false));
            System.out.println("probs= " + classifier.CalcLikelihoodNoLog(words.get(i),true) + " " + classifier.CalcLikelihoodNoLog(words.get(i),false));
            if(classifier.CalcLikelihoodLog(words.get(i),true) > classifier.CalcLikelihoodLog(words.get(i),false)) {
                spamOrHam = "spam";
                rightOrWrong = "right";
                numSpamCorrect++;
            }
            else {
                spamOrHam = "ham";
                rightOrWrong = "wrong";
            }
            totalEmails++;
            System.out.println("TEST "  + (i + 1) + " " + trueCnt + "/" + vocab.size() + " features true " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),true)) +
                    " " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),false)) + " "  + spamOrHam + " " + rightOrWrong);
        }
        System.out.println(numSpamCorrect + " out of " + words.size() +  " classified correctly.");

        words = classifier.parseTest(hamFile);
        System.out.println("Testing ham emails.");
        for(int i = 0; i < words.size(); i++) {
            System.out.println("Test email " + (i + 1));
            System.out.println("priors= " + priorSpam + " " + priorHam);
            System.out.println("likelihoods= " + classifier.CalcLikelihoodNoPrior(words.get(i),true) + " " + classifier.CalcLikelihoodNoPrior(words.get(i),false));
            System.out.println("probs= " + classifier.CalcLikelihoodNoLog(words.get(i),true) + " " + classifier.CalcLikelihoodNoLog(words.get(i),false));
            if(classifier.CalcLikelihoodLog(words.get(i),true) > classifier.CalcLikelihoodLog(words.get(i),false)) {
                spamOrHam = "spam";
                rightOrWrong = "wrong";
            }
            else {
                spamOrHam = "ham";
                numHamCorrect++;
                rightOrWrong = "right";
            }
            System.out.println("TEST "  + (i + 1) + " " + trueCnt + "/" + vocab.size() + " features true " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),true)) +
                    " " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),false)) + " "  + spamOrHam + " " + rightOrWrong);
            totalEmails++;
        }
        System.out.println(numHamCorrect + " out of " + words.size() + " classified correctly.");
        System.out.println("Total: " + (numSpamCorrect + numHamCorrect) + "/" + totalEmails + " emails classified correctly.");
    }

    private static void printNoDebugOutput(String spamFile, String hamFile) throws FileNotFoundException {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();

        List<Set<String>> words = classifier.parseTest(spamFile);
        int numSpamCorrect = 0;
        int numHamCorrect = 0;
        int totalEmails = 0;
        String rightOrWrong;
        for(int i = 0; i < words.size(); i++) {
            if(classifier.CalcLikelihoodLog(words.get(i),true) > classifier.CalcLikelihoodLog(words.get(i),false)) {
                spamOrHam = "spam";
                rightOrWrong = "right";
                numSpamCorrect++;
            }
            else {
                spamOrHam = "ham";
                rightOrWrong = "wrong";
            }
            totalEmails++;
            System.out.println("TEST "  + (i + 1) + " " + trueCnt + "/" + vocab.size() + " features true " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),true)) +
                    " " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),false)) + " "  + spamOrHam + " " + rightOrWrong);
        }

        words = classifier.parseTest(hamFile);
        for(int i = 0; i < words.size(); i++) {
            if(classifier.CalcLikelihoodLog(words.get(i),true) > classifier.CalcLikelihoodLog(words.get(i),false)) {
                spamOrHam = "spam";
                rightOrWrong = "wrong";
            }
            else {
                spamOrHam = "ham";
                numHamCorrect++;
                rightOrWrong = "right";
            }
            System.out.println("TEST "  + (i + 1) + " " + trueCnt + "/" + vocab.size() + " features true " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),true)) +
                    " " + String.format("%.3f",classifier.CalcLikelihoodLog(words.get(i),false)) + " "  + spamOrHam + " " + rightOrWrong);
            totalEmails++;


        }
        System.out.println("Total: " + (numSpamCorrect + numHamCorrect) + "/" + totalEmails + " emails classified correctly.");
    }
}
