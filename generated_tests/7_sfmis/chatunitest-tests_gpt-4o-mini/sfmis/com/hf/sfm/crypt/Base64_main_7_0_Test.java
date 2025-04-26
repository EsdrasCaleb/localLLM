package com.hf.sfm.crypt;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_main_7_0_Test {

    @Test
    public void testMain() {
        // Arrange
        String expectedOutput = "MDI0NTY3ODk=\n0123456789\n";
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outputStream));
        // Act
        // Empty args as the method does not use them
        String[] args = {};
        Base64.main(args);
        // Assert
        String actualOutput = outputStream.toString();
        assertEquals(expectedOutput, actualOutput);
        // Clean up
        System.setOut(originalOut);
    }
}
