package org.templateit.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Iterator;

public class DelimitedFileReader_next_1_2_Test {

    private DelimitedFileReader reader;

    private BufferedReader mockReader;

    private File tempFile;

    private Path tempFilePath;

    @BeforeEach
    void setUp() throws IOException {
        tempFilePath = Files.createTempFile("test", ".txt");
        tempFile = tempFilePath.toFile();
        tempFile.deleteOnExit();
        mockReader = Mockito.mock(BufferedReader.class);
    }

    @Test
    void testNext_HasNextLine() throws Exception {
        String testLine = "apple,banana,cherry";
        // Simulate one line then EOF
        when(mockReader.readLine()).thenReturn(testLine, null);
        // Use reflection to set private fields for testing
        Field readerField = DelimitedFileReader.class.getDeclaredField("reader");
        readerField.setAccessible(true);
        readerField.set(reader, mockReader);
        Field nextLineField = DelimitedFileReader.class.getDeclaredField("nextLine");
        nextLineField.setAccessible(true);
        nextLineField.set(reader, testLine);
        Field hasNextField = DelimitedFileReader.class.getDeclaredField("hasNext");
        hasNextField.setAccessible(true);
        hasNextField.set(reader, true);
        reader = new DelimitedFileReader(tempFile, ",");
        String[] result = reader.next();
        assertEquals(3, result.length);
        assertEquals("apple", result[0]);
        assertEquals("banana", result[1]);
        assertEquals("cherry", result[2]);
        verify(mockReader, times(1)).readLine();
    }

    @Test
    void testNext_NoNextLine() throws Exception {
        // Simulate EOF immediately
        when(mockReader.readLine()).thenReturn(null);
        // Use reflection to set private fields for testing
        Field readerField = DelimitedFileReader.class.getDeclaredField("reader");
        readerField.setAccessible(true);
        readerField.set(reader, mockReader);
        Field nextLineField = DelimitedFileReader.class.getDeclaredField("nextLine");
        nextLineField.setAccessible(true);
        nextLineField.set(reader, null);
        Field hasNextField = DelimitedFileReader.class.getDeclaredField("hasNext");
        hasNextField.setAccessible(true);
        hasNextField.set(reader, false);
        reader = new DelimitedFileReader(tempFile, ",");
        assertThrows(NoSuchElementException.class, () -> reader.next());
        // Ensure readLine is not called when no lines exist
        verify(mockReader, times(0)).readLine();
    }

    @Test
    void testNext_MultipleLines() throws Exception {
        when(mockReader.readLine()).thenReturn("one,two", "three,four", null);
        // Use reflection to set private fields for testing
        Field readerField = DelimitedFileReader.class.getDeclaredField("reader");
        readerField.setAccessible(true);
        readerField.set(reader, mockReader);
        Field nextLineField = DelimitedFileReader.class.getDeclaredField("nextLine");
        nextLineField.setAccessible(true);
        nextLineField.set(reader, "one,two");
        Field hasNextField = DelimitedFileReader.class.getDeclaredField("hasNext");
        hasNextField.setAccessible(true);
        hasNextField.set(reader, true);
        reader = new DelimitedFileReader(tempFile, ",");
        String[] result1 = reader.next();
        assertEquals(2, result1.length);
        assertEquals("one", result1[0]);
        assertEquals("two", result1[1]);
        String[] result2 = reader.next();
        assertEquals(2, result2.length);
        assertEquals("three", result2[0]);
        assertEquals("four", result2[1]);
        assertThrows(NoSuchElementException.class, () -> reader.next());
    }
}
