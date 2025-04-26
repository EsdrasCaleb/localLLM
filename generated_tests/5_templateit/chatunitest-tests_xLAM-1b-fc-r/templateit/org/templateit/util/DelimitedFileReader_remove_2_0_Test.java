package org.templateit.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.io.FileNotFoundException;

@ExtendWith(MockitoExtension.class)
class DelimitedFileReader_remove_2_0_Test {

    @Mock
    private DelimitedFileReader delimitedFileReader;

    @Test
    void remove() {
        doNothing().when(delimitedFileReader).remove();
        assertDoesNotThrow(() -> delimitedFileReader.remove());
    }
}
