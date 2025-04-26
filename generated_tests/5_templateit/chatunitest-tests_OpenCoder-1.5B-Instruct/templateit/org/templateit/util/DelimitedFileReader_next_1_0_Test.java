package org.templateit.util;

// Import necessary packages
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

// Source code of the JUnit class
public class DelimitedFileReader_next_1_0_Test {

    // Test method for the "next()" method
    @Test
    public void testNext() throws FileNotFoundException {
        // Create a mock object of the DelimitedFileReader class
        DelimitedFileReader mockReader = Mockito.mock(DelimitedFileReader.class);
        // Create a mock file
        File mockFile = Mockito.mock(File.class);
        // Create an array of strings representing the expected output
        String[] expectedOutput = { "name", "age", "city" };
        // Create an array of strings representing the actual output
        String[] actualOutput = null;
        // Setup the behavior of the mock object
        Mockito.when(mockReader.next()).thenReturn(expectedOutput);
        Mockito.when(mockReader.next()).thenReturn(null);
        // Call the "next()" method of the mock object
        actualOutput = mockReader.next();
        // Verify that the "next()" method returns the expected output
        Assertions.assertArrayEquals(expectedOutput, actualOutput);
        // Call the "next()" method again
        actualOutput = mockReader.next();
        // Verify that the "next()" method returns null
        Assertions.assertNull(actualOutput);
    }
}
