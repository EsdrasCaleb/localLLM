package com.hf.sfm.crypt;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_main_7_0_Test {

    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();

    private final PrintStream originalOut = System.out;

    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
    }

    @Test
    public void testMain() throws Exception {
        // Invoke the main method using reflection
        Method mainMethod = Base64.class.getDeclaredMethod("main", String[].class);
        mainMethod.invoke(null, (Object) new String[0]);
        // Expected output after encoding and decoding
        String expectedOutput = "MDEyMzQ1Njc4OQ==\n0123456789\n";
        // Verify the output
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    public void testByteArrayToBase64() throws Exception {
        // Access the private method using reflection
        Method byteArrayToBase64Method = Base64.class.getDeclaredMethod("byteArrayToBase64", byte[].class);
        byteArrayToBase64Method.setAccessible(true);
        // Test data
        byte[] testData = "0123456789".getBytes();
        // Invoke the method
        String result = (String) byteArrayToBase64Method.invoke(null, testData);
        // Expected result
        String expectedResult = "MDEyMzQ1Njc4OQ==";
        // Verify the result
        assertEquals(expectedResult, result);
    }

    @Test
    public void testBase64ToByteArray() throws Exception {
        // Access the private method using reflection
        Method base64ToByteArrayMethod = Base64.class.getDeclaredMethod("base64ToByteArray", String.class);
        base64ToByteArrayMethod.setAccessible(true);
        // Test data
        String testData = "MDEyMzQ1Njc4OQ==";
        // Invoke the method
        byte[] result = (byte[]) base64ToByteArrayMethod.invoke(null, testData);
        // Expected result
        byte[] expectedResult = "0123456789".getBytes();
        // Verify the result
        assertArrayEquals(expectedResult, result);
    }
}
