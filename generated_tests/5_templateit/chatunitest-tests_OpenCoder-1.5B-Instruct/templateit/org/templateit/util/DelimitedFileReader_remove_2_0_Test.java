package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Iterator;
import java.util.NoSuchElementException;

@ExtendWith(MockitoExtension.class)
class DelimitedFileReader_remove_2_0_Test {

    @Mock
    private BufferedReader reader;

    @InjectMocks
    private DelimitedFileReader delimitedFileReader;

    @Test
    void testRemove() throws IOException {
        String input = "line1\nline2\n";
        InputStream in = new ByteArrayInputStream(input.getBytes());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        System.setIn(in);
        System.setOut(new PrintStream(out));
        // read first line
        delimitedFileReader.next();
        // remove first line
        delimitedFileReader.remove();
        assertEquals("line2\n", out.toString());
        // read second line
        delimitedFileReader.next();
        // remove second line
        delimitedFileReader.remove();
        assertEquals("line2\n", out.toString());
    }
}
