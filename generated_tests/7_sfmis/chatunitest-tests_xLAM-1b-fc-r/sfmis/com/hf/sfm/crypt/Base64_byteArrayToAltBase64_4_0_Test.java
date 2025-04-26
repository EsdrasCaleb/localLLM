package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToAltBase64_4_0_Test {

    @Test
    public void byteArrayToAltBase64Test() {
        byte[] input = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
        String expectedOutput = "YWVzLXNlc3Rvcnk=";
        String result = Base64.byteArrayToAltBase64(input);
        assertEquals(expectedOutput, result);
    }
}
