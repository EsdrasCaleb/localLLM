package org.templateit.util;

import org.junit.jupiter.api.function.Executable;
import java.io.*;
import java.lang.reflect.Field;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Iterator;

public class DelimitedFileReader_remove_2_2_Test {

    @Test
    void testRemove() {
        DelimitedFileReader reader = createDelimitedFileReader();
        assertThrows(UnsupportedOperationException.class, () -> reader.remove());
    }

    private DelimitedFileReader createDelimitedFileReader() {
        try {
            File tempFile = File.createTempFile("test", ".txt");
            tempFile.deleteOnExit();
            return new DelimitedFileReader(tempFile, ",");
        } catch (IOException e) {
            fail("Failed to create temporary file: " + e.getMessage());
            return null;
        }
    }

    @Test
    void testNextWithEmptyFile() throws IOException, NoSuchElementException {
        File tempFile = File.createTempFile("test", ".txt");
        tempFile.deleteOnExit();
        DelimitedFileReader reader = new DelimitedFileReader(tempFile, ",");
        assertFalse(reader.hasNext());
        assertThrows(NoSuchElementException.class, () -> reader.next());
    }

    @Test
    void testNextWithOneLine() throws IOException, NoSuchElementException {
        File tempFile = File.createTempFile("test", ".txt");
        tempFile.deleteOnExit();
        String line = "test,data";
        try (FileWriter writer = new FileWriter(tempFile)) {
            writer.write(line);
        }
        DelimitedFileReader reader = new DelimitedFileReader(tempFile, ",");
        assertTrue(reader.hasNext());
        String[] next = reader.next();
        assertArrayEquals(new String[] { "test", "data" }, next);
        assertFalse(reader.hasNext());
        assertThrows(NoSuchElementException.class, () -> reader.next());
    }

    @Test
    void testNextWithMultipleLines() throws IOException, NoSuchElementException {
        File tempFile = File.createTempFile("test", ".txt");
        tempFile.deleteOnExit();
        String line1 = "test1,data1";
        String line2 = "test2,data2";
        try (FileWriter writer = new FileWriter(tempFile)) {
            writer.write(line1 + "\n");
            writer.write(line2);
        }
        DelimitedFileReader reader = new DelimitedFileReader(tempFile, ",");
        assertTrue(reader.hasNext());
        String[] next1 = reader.next();
        assertArrayEquals(new String[] { "test1", "data1" }, next1);
        assertTrue(reader.hasNext());
        String[] next2 = reader.next();
        assertArrayEquals(new String[] { "test2", "data2" }, next2);
        assertFalse(reader.hasNext());
        assertThrows(NoSuchElementException.class, () -> reader.next());
    }

    @Test
    void testHasNextWithIOException() throws IOException, NoSuchElementException, IllegalAccessException {
        File tempFile = File.createTempFile("test", ".txt");
        tempFile.deleteOnExit();
        try {
            DelimitedFileReader reader = new DelimitedFileReader(tempFile, ",");
            // Simulate IOException -  Not ideal testing, but demonstrates exception handling
            Field readerField = DelimitedFileReader.class.getDeclaredField("reader");
            readerField.setAccessible(true);
            readerField.set(reader, new BufferedReader(new FileReader(tempFile)) {

                @Override
                public String readLine() throws IOException {
                    throw new IOException("Simulated IOException");
                }
            });
            assertFalse(reader.hasNext());
        } catch (NoSuchFieldException e) {
            fail("Failed to access reader field: " + e.getMessage());
        }
    }
}
