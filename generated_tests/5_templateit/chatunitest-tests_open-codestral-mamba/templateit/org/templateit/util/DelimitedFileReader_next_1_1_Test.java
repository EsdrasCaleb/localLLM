package org.templateit.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileReader;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class DelimitedFileReader_next_1_1_Test {

    @InjectMocks
    private DelimitedFileReader delimitedFileReader;

    @Mock
    private BufferedReader bufferedReader;

    @BeforeEach
    void setUp() throws FileNotFoundException {
        // Use reflection to set the private field 'reader'
        try {
            java.lang.reflect.Field field = DelimitedFileReader.class.getDeclaredField("reader");
            field.setAccessible(true);
            field.set(delimitedFileReader, bufferedReader);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    void testNext() throws IOException {
        // Mock the behavior of bufferedReader.readLine()
        when(bufferedReader.readLine()).thenReturn("Hello,World");
        // Fix the bug by converting the result to a string array
        String[] result = delimitedFileReader.next();
        assertArrayEquals(new String[] { "Hello", "World" }, result);
    }

    @Test
    void testNext_EndOfFile() throws IOException {
        // Mock the behavior of bufferedReader.readLine() to return null
        when(bufferedReader.readLine()).thenReturn(null);
        // Call the next() method and assert that it throws NoSuchElementException
        assertThrows(NoSuchElementException.class, () -> {
            delimitedFileReader.next();
        });
    }

    @Test
    void testNext_IOException() throws IOException {
        // Mock the behavior of bufferedReader.readLine() to throw IOException
        when(bufferedReader.readLine()).thenThrow(IOException.class);
        // Call the next() method and assert that it throws IOException
        assertThrows(IOException.class, () -> {
            delimitedFileReader.next();
        });
    }

    @Test
    void next() throws IOException {
        when(bufferedReader.readLine()).thenReturn("a,b,c").thenReturn("d,e,f").thenReturn(null);
        // First call to next()
        delimitedFileReader.next();
        String[] next = delimitedFileReader.next();
        assertArrayEquals(new String[] { "a", "b", "c" }, next);
        next = delimitedFileReader.next();
        assertArrayEquals(new String[] { "d", "e", "f" }, next);
        assertThrows(NoSuchElementException.class, delimitedFileReader::next);
    }
}
