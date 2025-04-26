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

// Add this annotation to enable Mockito
@ExtendWith(MockitoExtension.class)
public class DelimitedFileReader_hasNext_0_3_Test {

    @Mock
    private BufferedReader reader;

    @InjectMocks
    private DelimitedFileReader delimitedFileReader;

    @BeforeEach
    public void setUp() throws FileNotFoundException {
        MockitoAnnotations.initMocks(this);
        delimitedFileReader = new DelimitedFileReader(new File("test.txt"));
    }

    @Test
    public void testHasNext() throws IOException {
        // Test when the file has content
        when(reader.ready()).thenReturn(true);
        assertTrue(delimitedFileReader.hasNext());
        // Test when the file is empty
        when(reader.ready()).thenReturn(false);
        assertFalse(delimitedFileReader.hasNext());
        // Test when an IOException occurs
        when(reader.ready()).thenThrow(new IOException());
        assertFalse(delimitedFileReader.hasNext());
    }
}
