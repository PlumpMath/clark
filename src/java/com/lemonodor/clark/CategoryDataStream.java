package com.lemonodor.clark;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.util.ObjectStream;

public class CategoryDataStream implements ObjectStream<DocumentSample> {

  File[] inputFiles;
  int inputFilesIndex = 0;

  String encoding;
  BufferedReader reader;
  Tokenizer tokenizer;

  String line;
  File currentFile;

  public CategoryDataStream(File[] inputFiles, Tokenizer tokenizer) {
    this(inputFiles, "UTF-8", tokenizer);
  }

  public CategoryDataStream(File[] inputFiles, String encoding, Tokenizer tokenizer) {
    this.inputFiles = inputFiles;
    this.encoding   = encoding;

    if (tokenizer == null) {
      this.tokenizer = SimpleTokenizer.INSTANCE;
    }
    else {
      this.tokenizer = tokenizer;
    }
  }

  public CategoryDataStream(String fileName, Tokenizer tokenizer) {
    this(fileName, "UTF-8", tokenizer);
  }

  public CategoryDataStream(String fileName, String encoding, Tokenizer tokenizer) {
    this((File[]) null, encoding, tokenizer);

    inputFiles = new File[1];
    inputFiles[0] = new File(fileName);
  }

  public void reset() {
    close();
    line = null;
    inputFilesIndex = 0;
  }

  public void close() {
    if (reader != null) {
      try {
        reader.close();
      }
      catch (IOException ex) {
          System.err.println("IOException on close" + ex);
      }
    }
  }

  /** Set the current buffered line to null, and attempt to obtain the next
   *  line of training data, if we run out of lines in one file, move on to
   *  the next. If we are out of files, line will remain null when this
   *  method returns.
   *
   *  @throws RuntimeException if there's a problem reading any of the input
   *    files.
   */
  protected void getNextLine() {
    line = null;

    try {
      while (line == null) {
        if (reader == null) {
          // no more files to read;
          if (inputFilesIndex >= inputFiles.length) break;

          // open the next file.
          currentFile = inputFiles[inputFilesIndex];
          System.err.println("Opening " + currentFile);
          reader = new BufferedReader(
              new InputStreamReader(
                  new FileInputStream(currentFile),
                  encoding));
        }

        line = reader.readLine();
        System.err.println("Read line: '" + line + "'");
        if (line == null) {
          // done with this reader, move to the next file.
          reader = null;
          inputFilesIndex++;
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException("Error reading input from: " + currentFile, e);
    }
  }



  public boolean hasNext() {
    getNextLine();
    return line != null;
  }

//<start id="maxent.examples.train.event"/>
  public DocumentSample read() {
    if (line == null && !hasNext()) { //<co id="mee.train.read"/>
      return null;
    }
    int split = line.indexOf('\t'); //<co id="mee.train.cat"/>
    if (split < 0)
      throw new RuntimeException("Invalid line in "
          + inputFiles[inputFilesIndex]);
    String category = line.substring(0,split);
    String document = line.substring(split+1);
    System.err.println("cat:" + category + " doc:" + document);
    line = null; // mark line as consumed
    String[] tokens = tokenizer.tokenize(document); //<co id="mee.train.tok"/>
    return new DocumentSample(category, tokens); //<co id="mee.train.sample"/>
  }
  /*<calloutlist>
  <callout arearefs="mee.train.read">Read a line training data</callout>
  <callout arearefs="mee.train.cat">Extract category</callout>
  <callout arearefs="mee.train.tok">Tokenize content</callout>
  <callout arearefs="mee.train.tok">Create sample</callout>
  </calloutlist>*/
//<end id="maxent.examples.train.event"/>
}
