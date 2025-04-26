package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToAltBase64_4_3_Test {

    @Test
    public void testByteArrayToAltBase64() {
        // Test data
        byte[] input = { 1, 2, 3, 4, 5 };
        String expected = "ABCDEF";
        // Call the method under test
        String actual = Base64.byteArrayToAltBase64(input);
        // Assert the result
        assertEquals(expected, actual);
    }
}
