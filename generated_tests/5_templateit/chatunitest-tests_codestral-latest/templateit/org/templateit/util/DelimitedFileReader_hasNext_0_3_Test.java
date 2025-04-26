package org.templateit.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileReader;
import java.util.Iterator;
import java.util.NoSuchElementException;

@ExtendWith(MockitoExtension.class)
class DelimitedFileReader_hasNext_0_3_Test {

    @Mock
    private BufferedReader reader;

    @InjectMocks
    private DelimitedFileReader delimitedFileReader;

    @BeforeEach
    void setUp() throws FileNotFoundException, NoSuchFieldException, IllegalAccessException {
        delimitedFileReader = new DelimitedFileReader(new File("test.txt"), ",");
        Field readerField = DelimitedFileReader.class.getDeclaredField("reader");
        readerField.setAccessible(true);
        readerField.set(delimitedFileReader, reader);
    }

    @Test
    void testHasNext_WhenNextLineIsNullAndReaderReturnsLine() throws IOException {
        when(reader.readLine()).thenReturn("line1");
        assertTrue(delimitedFileReader.hasNext());
    }

    @Test
    void testHasNext_WhenNextLineIsNullAndReaderReturnsNull() throws IOException {
        when(reader.readLine()).thenReturn(null);
        assertFalse(delimitedFileReader.hasNext());
    }

    @Test
    void testHasNext_WhenNextLineIsNotNull() throws IOException {
        when(reader.readLine()).thenReturn("line1");
        assertTrue(delimitedFileReader.hasNext());
        assertTrue(delimitedFileReader.hasNext());
    }

    @Test
    void testHasNext_WhenHasNextIsFalse() throws IOException {
        when(reader.readLine()).thenReturn(null);
        assertFalse(delimitedFileReader.hasNext());
        assertFalse(delimitedFileReader.hasNext());
    }

    @Test
    void testHasNext_WhenIOExceptionOccurs() throws IOException {
        when(reader.readLine()).thenThrow(new IOException());
        assertFalse(delimitedFileReader.hasNext());
    }
}
