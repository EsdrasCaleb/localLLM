package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileReader;
import java.util.Iterator;

@ExtendWith(MockitoExtension.class)
public class DelimitedFileReader_remove_2_4_Test {

    @Mock
    private BufferedReader mockReader;

    @InjectMocks
    private DelimitedFileReader testObject;

    @Test
    public void testRemove() throws IOException {
        // Given
        File file = new File("test.txt");
        String delimiter = ",";
        when(mockReader.readLine()).thenReturn("line1,line2,line3");
        testObject = new DelimitedFileReader(file, delimiter);
        // When & Then
        assertThrows(UnsupportedOperationException.class, () -> testObject.remove());
    }
}
